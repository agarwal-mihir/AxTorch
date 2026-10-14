// Shared-memory LUT GEMM for approximate int8 multipliers.
//
//   C[z, zz, m, l] = epilogue( sum_k T[ia(A[z, m, k]) + ib(B[z, zz, k, l])] )
//
// n = zz * L + l folds a second batch dimension into N (images in a conv), so
// 16/32-channel layers still fill whole tiles. T is a 32768-entry int16 view of
// the 256x256 LUT (64 KB, built by approxtorch.backend.lut_kernel.table); ia/ib are byte offsets:
//   MODE 0 (conv):    A = weight [-128,127], B = activation [0,127], T[w+128][x]
//   MODE 1 (linear):  A = activation [0,127], B = weight [-128,127], T[x][w+128]
//   MODE 2 (signmag): both in [-127,127], T = [M; -M] with M[|a|][|b|]; the two
//                     sign bits add at bit 15 and & 0xFFFF drops the carry (xor).
// Integer sums are exact, so results match ApproxTorch bit for bit. The float
// epilogue uses _rn intrinsics (no FMA contraction) to reproduce PyTorch's
// ((float(acc) + c) * s1) * s2 + bias rounding exactly.

#include <algorithm>
#include <optional>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

// BK = 32: each __syncthreads drains the lookup pipeline, so fewer barriers per k matter more
// than tile size. B tiles stay raw int8 in shared memory (offsets formed in the inner loop,
// which has spare integer issue slots), leaving room for the im2col table.
constexpr int BK = 32, TM = 4, TN = 4, TABLE_BYTES = 65536;
constexpr int MAX_IM2COL_K = 5632;  // ktab in shared memory: 96 KB with the 16x256 tile (V100 limit)

struct Params {
  const int8_t* A;
  const int8_t* B;
  const int16_t* T;
  void* C;
  int Z, M, N, K, L;
  long long sAz, sAm, sAk, sBz, sBzz, sBk, sBl, sCz, sCzz, sCm, sCl;
  const float* c;  // nullptr: raw int32 output
  const float* s1;
  const float* s2;    // optional
  const float* bias;  // optional
  bool vec_on_n;      // epilogue vectors indexed by n (linear) instead of m (conv)
  bool cvec;          // 16-byte C stores: 4 consecutive n contiguous and aligned
  // LOAD_IM2COL: B is the zero-padded int8 NCHW input (batch stride sBzz, row length W),
  // gathered in F.unfold order; ktab[k] = c*H*W + kh*dh*W + kw*dw.
  const int* ktab;
  int W, OW, sh, sw;
};

enum { LOAD_STRIDED, LOAD_VEC, LOAD_IM2COL };

// Out-of-range operands are masked into the table (memory safety only; callers
// guarantee the ranges above).
template <int MODE>
__device__ __forceinline__ int a_offset(int a) {
  if (MODE == 0) return (a + 128) << 8;
  if (MODE == 1) return (a & 0x7F) << 9;
  return ((a < 0 ? -a : a) << 8 & 0x7FFF) | ((a < 0) << 15);
}

template <int MODE>
__device__ __forceinline__ int b_offset(int b) {
  if (MODE == 0) return (b & 0x7F) << 1;
  if (MODE == 1) return (b + 128) << 1;
  return ((b < 0 ? -b : b) << 1 & 0xFF) | ((b < 0) << 15);
}

template <int MODE, int BM, int BN, int THREADS, int LOAD>
__global__ void __launch_bounds__(THREADS) lut_gemm_kernel(const Params p) {
  static_assert((BM / TM) * (BN / TN) == THREADS && THREADS % BN == 0, "bad tile config");
  constexpr int A_PER = (BK * BM + THREADS - 1) / THREADS;
  constexpr int B_PER = BK * BN / THREADS, B_KSTEP = THREADS / BN;  // scalar B: one column per thread
  constexpr bool VEC = LOAD == LOAD_VEC;
  constexpr int V_PER = (BK * BN / 16 + THREADS - 1) / THREADS;  // VEC: 16-byte loads per thread
  extern __shared__ __align__(16) unsigned char smem[];
  int* As = reinterpret_cast<int*>(smem + TABLE_BYTES);  // [BK][BM] byte offsets
  int8_t* Bs = reinterpret_cast<int8_t*>(As + BK * BM);  // [BK][BN] raw operands
  int* Ks = reinterpret_cast<int*>(Bs + BK * BN);        // [K] im2col offsets

  const int tid = threadIdx.x;
  for (int i = tid; i < TABLE_BYTES / 16; i += THREADS)
    reinterpret_cast<int4*>(smem)[i] = reinterpret_cast<const int4*>(p.T)[i];
  if (LOAD == LOAD_IM2COL) {
    for (int i = tid; i < p.K; i += THREADS) Ks[i] = p.ktab[i];
    __syncthreads();
  }

  const int tilesM = (p.M + BM - 1) / BM, tilesN = (p.N + BN - 1) / BN;
  const long long tiles = (long long)p.Z * tilesM * tilesN;
  const int tx = tid % (BN / TN), ty = tid / (BN / TN);
  // VEC: each thread loads 16 consecutive n (one image row, 16-byte aligned) of one k, at
  // k = b_kk + r * (THREADS / (BN / 16)).
  const int b_nn = VEC ? tid % (BN / 16) * 16 : tid % BN;
  const int b_kk = VEC ? tid / (BN / 16) : tid / BN;
  constexpr int V_KSTEP = THREADS / (BN / 16);

  int a_reg[A_PER];
  int b_reg[VEC ? 1 : B_PER];
  int4 b_vec[VEC ? V_PER : 1];

  // Persistent blocks: the 64 KB table is loaded once per block, not per tile.
  for (long long t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int tn = (int)(t % tilesN);
    const long long rest = t / tilesN;
    const int tm = (int)(rest % tilesM), z = (int)(rest / tilesM);
    const int m0 = tm * BM, n0 = tn * BN;
    const int8_t* Az = p.A + z * p.sAz;
    long long b_off = -1;
    const bool b_valid = n0 + b_nn < p.N;
    if (b_valid) {
      const int n = n0 + b_nn, image = n / p.L, l = n % p.L;
      if (LOAD == LOAD_IM2COL) {
        b_off = image * p.sBzz + (long long)(l / p.OW * p.sh) * p.W + l % p.OW * p.sw;
      } else {
        b_off = z * p.sBz + (long long)image * p.sBzz + (long long)l * p.sBl;
      }
    }

    auto load = [&](int k0) {
#pragma unroll
      for (int r = 0; r < A_PER; ++r) {
        const int e = tid + r * THREADS, m = m0 + e / BK, k = k0 + e % BK;
        a_reg[r] = (e < BK * BM && m < p.M && k < p.K) ? Az[m * p.sAm + k * p.sAk] : 0;
      }
      if (VEC) {
#pragma unroll
        for (int r = 0; r < V_PER; ++r) {
          const int kk = b_kk + r * V_KSTEP, k = k0 + kk;
          b_vec[r] = (kk < BK && b_valid && k < p.K)
                         ? *reinterpret_cast<const int4*>(p.B + b_off + k * p.sBk)
                         : make_int4(0, 0, 0, 0);
        }
      } else {
#pragma unroll
        for (int r = 0; r < B_PER; ++r) {
          const int k = k0 + b_kk + r * B_KSTEP;
          int b = 0;
          if (b_valid && k < p.K) {
            b = p.B[b_off + (LOAD == LOAD_IM2COL ? Ks[k] : k * p.sBk)];
          }
          b_reg[VEC ? 0 : r] = b;
        }
      }
    };
    auto store = [&]() {
#pragma unroll
      for (int r = 0; r < A_PER; ++r) {
        const int e = tid + r * THREADS;
        if (e < BK * BM) As[e % BK * BM + e / BK] = a_offset<MODE>(a_reg[r]);
      }
      if (VEC) {
#pragma unroll
        for (int r = 0; r < V_PER; ++r) {
          const int kk = b_kk + r * V_KSTEP;
          if (kk < BK) *reinterpret_cast<int4*>(&Bs[kk * BN + b_nn]) = b_vec[r];
        }
      } else {
#pragma unroll
        for (int r = 0; r < B_PER; ++r) Bs[(b_kk + r * B_KSTEP) * BN + b_nn] = (int8_t)b_reg[VEC ? 0 : r];
      }
    };

    int acc[TM][TN] = {};
    load(0);
    store();
    __syncthreads();
    for (int k0 = 0; k0 < p.K; k0 += BK) {
      const bool more = k0 + BK < p.K;
      if (more) load(k0 + BK);  // global loads in flight during this block's lookups
      auto step = [&](int kk) {
        const int4 av = *reinterpret_cast<const int4*>(&As[kk * BM + ty * TM]);
        const unsigned bw = *reinterpret_cast<const unsigned*>(&Bs[kk * BN + tx * TN]);
        const int a4[4] = {av.x, av.y, av.z, av.w};
        const int b4[4] = {b_offset<MODE>((int8_t)bw), b_offset<MODE>((int8_t)(bw >> 8)),
                           b_offset<MODE>((int8_t)(bw >> 16)), b_offset<MODE>((int8_t)(bw >> 24))};
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            const int off = MODE == 2 ? (a4[i] + b4[j]) & 0xFFFF : a4[i] + b4[j];
            acc[i][j] += *reinterpret_cast<const int16_t*>(smem + off);
          }
      };
      if (more) {
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) step(kk);
      } else {
        for (int kk = 0; kk < p.K - k0; ++kk) step(kk);
      }
      __syncthreads();
      if (more) {
        store();
        __syncthreads();
      }
    }

    long long c_off[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int n = n0 + tx * TN + j;
      c_off[j] = n < p.N ? z * p.sCz + (long long)(n / p.L) * p.sCzz + (long long)(n % p.L) * p.sCl : -1;
      if (p.cvec) break;  // 4 consecutive n share an image: one 16-byte store
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int m = m0 + ty * TM + i;
      if (m >= p.M || c_off[0] < 0) continue;
      int raw[TN];
      float y[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        raw[j] = acc[i][j];
        if (p.c != nullptr) {
          const int v = p.vec_on_n ? n0 + tx * TN + j : m;
          y[j] = __fmul_rn(__fadd_rn(__int2float_rn(acc[i][j]), p.c[v]), p.s1[v]);
          if (p.s2 != nullptr) y[j] = __fmul_rn(y[j], p.s2[v]);
          if (p.bias != nullptr) y[j] = __fadd_rn(y[j], p.bias[v]);
        }
      }
      if (p.cvec) {
        const long long o = c_off[0] + m * p.sCm;
        if (p.c == nullptr) *reinterpret_cast<int4*>(static_cast<int*>(p.C) + o) = make_int4(raw[0], raw[1], raw[2], raw[3]);
        else *reinterpret_cast<float4*>(static_cast<float*>(p.C) + o) = make_float4(y[0], y[1], y[2], y[3]);
      } else {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          if (c_off[j] < 0) continue;
          const long long o = c_off[j] + m * p.sCm;
          if (p.c == nullptr) static_cast<int*>(p.C)[o] = raw[j];
          else static_cast<float*>(p.C)[o] = y[j];
        }
      }
    }
  }
}

template <int MODE, int BM, int BN, int THREADS, int LOAD>
void launch(const Params& p) {
  auto kernel = lut_gemm_kernel<MODE, BM, BN, THREADS, LOAD>;
  constexpr size_t base = TABLE_BYTES + BK * BM * 4 + BK * BN;
  constexpr size_t smem_max = base + (LOAD == LOAD_IM2COL ? MAX_IM2COL_K * 4 : 0);
  const size_t smem = base + (LOAD == LOAD_IM2COL ? p.K * 4 : 0);
  static int grid_cap = 0;  // ponytail: one device per process, as the evaluator pool runs
  if (grid_cap == 0) {
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max));
    int per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, THREADS, smem_max));
    grid_cap = at::cuda::getCurrentDeviceProperties()->multiProcessorCount * std::max(per_sm, 1);
  }
  const long long tiles = (long long)p.Z * ((p.M + BM - 1) / BM) * ((p.N + BN - 1) / BN);
  const int grid = (int)std::min<long long>(tiles, grid_cap);
  if (grid == 0) return;
  kernel<<<grid, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int MODE, int LOAD>
void dispatch(const Params& p) {
  // A warp shares one A row when BN / TN >= 32, which keeps table-bank conflicts low.
  if (p.M <= 16) launch<MODE, 16, 256, 256, LOAD>(p);
  else if (p.N <= 64) launch<MODE, 64, 64, 256, LOAD>(p);
  else launch<MODE, 32, 128, 256, LOAD>(p);
}

template <int MODE>
void dispatch(const Params& p, bool vec) {
  if (vec) dispatch<MODE, LOAD_VEC>(p);
  else dispatch<MODE, LOAD_STRIDED>(p);
}

bool aligned(const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; }

// torch.round(x / scale + zero).clamp(0, 127).to(torch.int8), rounding step for step, written
// into y = x zero-padded by (ph, pw) on its last two dims (H, W are x's).
__global__ void quantize_u7_kernel(const float* __restrict__ x, int8_t* __restrict__ y, int n,
                                   int H, int W, int ph, int pw, const float* scale,
                                   const float* zero) {
  const float s = *scale, zp = *zero;
  const int Hp = H + 2 * ph, Wp = W + 2 * pw;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const int w = i % Wp - pw, h = i / Wp % Hp - ph;
    float q = 0.f;
    if ((unsigned)h < (unsigned)H && (unsigned)w < (unsigned)W)
      q = fminf(fmaxf(rintf(__fadd_rn(__fdiv_rn(x[i / Wp / Hp * H * W + h * W + w], s), zp)), 0.f),
                127.f);
    y[i] = (int8_t)q;
  }
}

const float* vec(const std::optional<at::Tensor>& t, int64_t size) {
  if (!t.has_value()) return nullptr;
  TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kFloat && t->is_contiguous() &&
                  t->numel() >= size,
              "epilogue vectors must be contiguous float32 CUDA tensors");
  return t->data_ptr<float>();
}

}  // namespace

void lut_gemm(const at::Tensor& A, const at::Tensor& B, const at::Tensor& T, at::Tensor& C,
              int64_t mode, int64_t Z, int64_t M, int64_t N, int64_t K, int64_t L,
              std::vector<int64_t> sA, std::vector<int64_t> sB, std::vector<int64_t> sC,
              const std::optional<at::Tensor>& c, const std::optional<at::Tensor>& s1,
              const std::optional<at::Tensor>& s2, const std::optional<at::Tensor>& bias,
              bool vec_on_n) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && T.is_cuda() && C.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(A.scalar_type() == at::kChar && B.scalar_type() == at::kChar, "int8 operands required");
  TORCH_CHECK(T.scalar_type() == at::kShort && T.is_contiguous() && T.numel() * 2 == TABLE_BYTES,
              "table must be 32768 contiguous int16");
  TORCH_CHECK(mode >= 0 && mode <= 2 && sA.size() == 3 && sB.size() == 4 && sC.size() == 4);
  TORCH_CHECK(L > 0 && N % L == 0, "N must be a multiple of L");
  const bool raw = !c.has_value();
  TORCH_CHECK(C.scalar_type() == (raw ? at::kInt : at::kFloat), "C must be int32 (raw) or float32");
  TORCH_CHECK(raw || s1.has_value(), "float epilogue needs c and s1");
  const int64_t vlen = vec_on_n ? N : M;
  const c10::cuda::CUDAGuard guard(A.device());
  Params p{A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), T.data_ptr<int16_t>(), C.data_ptr(),
                 (int)Z, (int)M, (int)N, (int)K, (int)L,
                 sA[0], sA[1], sA[2], sB[0], sB[1], sB[2], sB[3], sC[0], sC[1], sC[2], sC[3],
                 raw ? nullptr : vec(c, vlen), raw ? nullptr : vec(s1, vlen),
                 raw ? nullptr : vec(s2, vlen), raw ? nullptr : vec(bias, vlen), vec_on_n};
  // 16-byte B loads and C stores: every 16 consecutive n are contiguous and aligned.
  p.cvec = L % 4 == 0 && sC[3] == 1 && sC[0] % 4 == 0 && sC[1] % 4 == 0 && sC[2] % 4 == 0 &&
           aligned(p.C);
  const bool vec = p.cvec && L % 16 == 0 && sB[3] == 1 && sB[0] % 16 == 0 && sB[1] % 16 == 0 &&
                   sB[2] % 16 == 0 && aligned(p.B);
  if (mode == 0) dispatch<0>(p, vec);
  else if (mode == 1) dispatch<1>(p, vec);
  else dispatch<2>(p, vec);
}

// Implicit-GEMM conv (mode 0): x int8 NCHW, already zero-padded; kernels [O, C*KH*KW]
// -> float [B, O, OH, OW].
at::Tensor lut_conv2d(const at::Tensor& x, const at::Tensor& kernels, const at::Tensor& T,
                      const at::Tensor& ktab, int64_t OH, int64_t OW, int64_t sh, int64_t sw,
                      const at::Tensor& c, const at::Tensor& s1,
                      const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kChar && x.dim() == 4 && x.is_contiguous());
  TORCH_CHECK(kernels.is_cuda() && kernels.scalar_type() == at::kChar && kernels.is_contiguous());
  TORCH_CHECK(T.scalar_type() == at::kShort && T.is_contiguous() && T.numel() * 2 == TABLE_BYTES);
  TORCH_CHECK(ktab.is_cuda() && ktab.scalar_type() == at::kInt && ktab.is_contiguous() &&
              ktab.numel() == kernels.size(1) && kernels.size(1) <= MAX_IM2COL_K);
  const int64_t B = x.size(0), O = kernels.size(0), K = kernels.size(1), L = OH * OW;
  TORCH_CHECK(x.stride(0) < INT32_MAX);
  auto C = at::empty({B, O, OH, OW}, x.options().dtype(at::kFloat));
  if (C.numel() == 0) return C;
  const c10::cuda::CUDAGuard guard(x.device());
  Params p{kernels.data_ptr<int8_t>(), x.data_ptr<int8_t>(), T.data_ptr<int16_t>(), C.data_ptr(),
           1, (int)O, (int)(B * L), (int)K, (int)L,
           0, K, 1, 0, x.stride(0), 0, 0, 0, O * L, L, 1,
           vec(c, O), vec(s1, O), nullptr, vec(bias, O), false};
  p.cvec = L % 4 == 0;
  p.ktab = ktab.data_ptr<int>();
  p.W = (int)x.size(3), p.OW = (int)OW, p.sh = (int)sh, p.sw = (int)sw;
  dispatch<0, LOAD_IM2COL>(p);
  return C;
}

at::Tensor quantize_u7(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& zero,
                       int64_t ph, int64_t pw) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat);
  TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == at::kFloat && zero.is_cuda() &&
              zero.scalar_type() == at::kFloat);
  TORCH_CHECK(x.dim() >= 2 || (ph == 0 && pw == 0));
  const auto xc = x.contiguous();
  auto sizes = xc.sizes().vec();
  const int H = x.dim() >= 2 ? (int)sizes.end()[-2] : 1, W = x.dim() >= 1 ? (int)sizes.back() : 1;
  if (x.dim() >= 2) sizes.end()[-2] += 2 * ph, sizes.back() += 2 * pw;
  auto y = at::empty(sizes, xc.options().dtype(at::kChar));
  TORCH_CHECK(y.numel() < INT32_MAX);
  const int n = (int)y.numel();
  if (n == 0 || xc.numel() == 0) return y.zero_();
  const c10::cuda::CUDAGuard guard(x.device());
  const int grid = std::min(n / 256 + 1, 65535);
  quantize_u7_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      xc.data_ptr<float>(), y.data_ptr<int8_t>(), n, H, W, (int)ph, (int)pw, scale.data_ptr<float>(),
      zero.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lut_gemm", &lut_gemm);
  m.def("lut_conv2d", &lut_conv2d);
  m.def("quantize_u7", &quantize_u7);
}
