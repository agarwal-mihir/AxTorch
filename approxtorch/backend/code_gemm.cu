// Generic LUT GEMM over 8/16-bit operand codes (int4, int8, uint8, int16, FP8, FP4, BF16, FP16):
//
//   C[z, m, n] = out( sum_k P(A[z, m, k], B[z, k, n]) ),  k ascending for every output
//
// A selects the table row. Tables are prepared by approxtorch.backend.lut_kernel (operand codes
// re-indexed so raw bits index them, transposed when the right operand selects the row):
//   I16/U16  8-bit codes, T[(a & mask) << shift | (b & mask)], 16-bit entries, exact int32 sums
//   F32      8-bit codes, float32 code-pair table, ordered float32 sums
//   F32SYM   FP8 with a sign-symmetric table: 128x128 magnitudes, sign = sa ^ sb
//   BF16     upstream RTL rule (csrc/float/approx_mul_bf16.cuh), 128x128 uint8 mantissa table
//   FP16     upstream RTL rule (approx_mul_fp16.cuh), 1024x1024 11-bit mantissa table (uint16)
//   INT16    sign-magnitude, |x| = 256 h + l: four lookups in a 256x256 uint16 table,
//            P = s * (T[ha][hb] << 16 + (T[ha][lb] + T[la][hb]) << 8 + T[la][lb]), exact int64
// Float kinds add products with __fadd_rn from +0 in ascending k (no FMA contraction), as the
// upstream FP16/BF16 kernels do, so results match them bit for bit.
// The table lives in shared memory when it fits next to the tiles, else it is read through L1.

#include <algorithm>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int BK = 32, THREADS = 256, MAX_DEVICES = 64;
enum Kind { K_I16, K_U16, K_F32, K_F32SYM, K_BF16, K_FP16, K_INT16, N_KINDS };

struct Params {
  const void* A;
  const void* B;
  const void* T;
  void* C;
  int Z, M, N, K, shift, tab_bytes;
  long long sAz, sAm, sAk, sBz, sBk, sBn;
};

template <class In> struct Vec4 { using T = uchar4; };
template <> struct Vec4<uint16_t> { using T = ushort4; };

// N consecutive shared-memory codes; 4 at once as one vector load.
template <int N, class In>
__device__ __forceinline__ void read(const In* s, In (&v)[N]) {
  if constexpr (N == 4) {
    const auto x = *reinterpret_cast<const typename Vec4<In>::T*>(s);
    v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w;
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = s[i];
  }
}

struct Code { unsigned i; };
struct Rtl { unsigned i; int e; unsigned s, m; };
struct SignMag { unsigned h, l; int s; };
struct IntAcc { int v = 0; };
struct FloatAcc { float v = 0.f; };
struct Int16Acc { int hh = 0, mid = 0, ll = 0; };

template <class Tab, bool SMEM>
struct Lut {
  const Tab* p;
  __device__ __forceinline__ Tab operator[](unsigned i) const {
    if constexpr (SMEM) return p[i];
    else return __ldg(p + i);
  }
};

template <int KIND> struct Pol;

template <class TabT, class AccT>
struct CodePol {
  using In = uint8_t; using Tab = TabT; using Op = Code; using Acc = AccT;
  static __device__ __forceinline__ Op pa(In a, int s) { return {(a & ((1u << s) - 1)) << s}; }
  static __device__ __forceinline__ Op pb(In b, int s) { return {b & ((1u << s) - 1)}; }
};

template <> struct Pol<K_I16> : CodePol<int16_t, IntAcc> {
  template <class L> static __device__ __forceinline__ void mac(Acc& c, Op a, Op b, L t) { c.v += t[a.i + b.i]; }
  static __device__ __forceinline__ void out(void* C, long long o, const Acc& c) { static_cast<int*>(C)[o] = c.v; }
};

template <> struct Pol<K_U16> : CodePol<uint16_t, IntAcc> {
  template <class L> static __device__ __forceinline__ void mac(Acc& c, Op a, Op b, L t) { c.v += t[a.i + b.i]; }
  static __device__ __forceinline__ void out(void* C, long long o, const Acc& c) { static_cast<int*>(C)[o] = c.v; }
};

template <> struct Pol<K_F32> : CodePol<float, FloatAcc> {
  template <class L> static __device__ __forceinline__ void mac(Acc& c, Op a, Op b, L t) { c.v = __fadd_rn(c.v, t[a.i + b.i]); }
  static __device__ __forceinline__ void out(void* C, long long o, const Acc& c) { static_cast<float*>(C)[o] = c.v; }
};

// Magnitude index in bits 13:0, sign in bit 31.
template <> struct Pol<K_F32SYM> : CodePol<float, FloatAcc> {
  static __device__ __forceinline__ Op pa(In a, int) { return {(a & 0x7Fu) << 7 | (a & 0x80u) << 24}; }
  static __device__ __forceinline__ Op pb(In b, int) { return {(b & 0x7Fu) | (b & 0x80u) << 24}; }
  template <class L> static __device__ __forceinline__ void mac(Acc& c, Op a, Op b, L t) {
    const unsigned bits = __float_as_uint(t[(a.i | b.i) & 0x3FFFu]) ^ ((a.i ^ b.i) & 0x80000000u);
    c.v = __fadd_rn(c.v, __uint_as_float(bits));
  }
  static __device__ __forceinline__ void out(void* C, long long o, const Acc& c) { static_cast<float*>(C)[o] = c.v; }
};

// Upstream RTL product: with entry = norm << F | frac, ((Ea + Eb - bias + norm) << F | frac) is
// (Ea << F) + (Eb << F) - (bias << F) + entry modulo 2^15, which wraps the exponent like the RTL.
// Either operand zero (+-0) gives +0.
template <> struct Pol<K_BF16> {
  using In = uint16_t; using Tab = uint8_t; using Op = Rtl; using Acc = FloatAcc;
  static __device__ __forceinline__ Op pa(In a, int) {
    return {(a & 0x7Fu) << 7, (int)(a & 0x7F80u) - 0x3F80, (a & 0x8000u) << 16, (a & 0x7FFFu) ? ~0u : 0u};
  }
  static __device__ __forceinline__ Op pb(In b, int) {
    return {b & 0x7Fu, (int)(b & 0x7F80u), (b & 0x8000u) << 16, (b & 0x7FFFu) ? ~0u : 0u};
  }
  template <class L> static __device__ __forceinline__ void mac(Acc& c, Op a, Op b, L t) {
    const unsigned mag = (unsigned)(a.e + b.e + (int)t[a.i + b.i]) & 0x7FFFu;
    c.v = __fadd_rn(c.v, __uint_as_float((mag << 16 | (a.s ^ b.s)) & (a.m & b.m)));
  }
  static __device__ __forceinline__ void out(void* C, long long o, const Acc& c) {
    static_cast<unsigned short*>(C)[o] = __bfloat16_as_ushort(__float2bfloat16_rn(c.v));
  }
};

template <> struct Pol<K_FP16> {
  using In = uint16_t; using Tab = uint16_t; using Op = Rtl; using Acc = FloatAcc;
  static __device__ __forceinline__ Op pa(In a, int) {
    return {(a & 0x3FFu) << 10, (int)(a & 0x7C00u) - (15 << 10), a & 0x8000u, (a & 0x7FFFu) ? 0xFFFFu : 0u};
  }
  static __device__ __forceinline__ Op pb(In b, int) {
    return {b & 0x3FFu, (int)(b & 0x7C00u), b & 0x8000u, (b & 0x7FFFu) ? 0xFFFFu : 0u};
  }
  template <class L> static __device__ __forceinline__ void mac(Acc& c, Op a, Op b, L t) {
    const unsigned mag = (unsigned)(a.e + b.e + (int)t[a.i + b.i]) & 0x7FFFu;
    const unsigned short h = (unsigned short)((mag | (a.s ^ b.s)) & (a.m & b.m));
    c.v = __fadd_rn(c.v, __half2float(__ushort_as_half(h)));
  }
  static __device__ __forceinline__ void out(void* C, long long o, const Acc& c) {
    static_cast<unsigned short*>(C)[o] = __half_as_ushort(__float2half_rn(c.v));
  }
};

template <> struct Pol<K_INT16> {
  using In = uint16_t; using Tab = uint16_t; using Op = SignMag; using Acc = Int16Acc;
  static __device__ __forceinline__ Op pa(In a, int) {
    const int v = (int16_t)a;
    const unsigned m = v < 0 ? -v : v;
    return {m >> 8 << 8, (m & 0xFF) << 8, v < 0 ? -1 : 0};
  }
  static __device__ __forceinline__ Op pb(In b, int) {
    const int v = (int16_t)b;
    const unsigned m = v < 0 ? -v : v;
    return {m >> 8, m & 0xFF, v < 0 ? -1 : 0};
  }
  template <class L> static __device__ __forceinline__ void mac(Acc& c, Op a, Op b, L t) {
    const int s = a.s ^ b.s;  // 0 or -1: (x ^ s) - s negates when the signs differ
    const int hh = t[a.h + b.h], mid = t[a.h + b.l] + t[a.l + b.h], ll = t[a.l + b.l];
    c.hh += (hh ^ s) - s;
    c.mid += (mid ^ s) - s;
    c.ll += (ll ^ s) - s;
  }
  static __device__ __forceinline__ void out(void* C, long long o, const Acc& c) {
    static_cast<long long*>(C)[o] = (long long)c.hh * 65536 + (long long)c.mid * 256 + c.ll;
  }
};

template <int KIND, int BM, int BN, int TM, int TN, bool SMEM>
__global__ void __launch_bounds__(THREADS) code_gemm_kernel(const Params p) {
  using P = Pol<KIND>;
  using In = typename P::In;
  static_assert((BM / TM) * (BN / TN) == THREADS && THREADS % BN == 0, "bad tile config");
  constexpr int A_PER = BK * BM / THREADS, B_PER = BK * BN / THREADS, B_KSTEP = THREADS / BN;
  extern __shared__ __align__(16) unsigned char smem[];
  In* As = reinterpret_cast<In*>(smem + (SMEM ? p.tab_bytes : 0));  // [BK][BM]
  In* Bs = As + BK * BM;                                            // [BK][BN]
  const int tid = threadIdx.x;
  if (SMEM) {
    for (int i = tid; i < p.tab_bytes / 16; i += THREADS)
      reinterpret_cast<int4*>(smem)[i] = reinterpret_cast<const int4*>(p.T)[i];
  }
  const Lut<typename P::Tab, SMEM> t{SMEM ? reinterpret_cast<const typename P::Tab*>(smem)
                                          : static_cast<const typename P::Tab*>(p.T)};

  const int tilesM = (p.M + BM - 1) / BM, tilesN = (p.N + BN - 1) / BN;
  const long long tiles = (long long)p.Z * tilesM * tilesN;
  const int tx = tid % (BN / TN), ty = tid / (BN / TN);  // a warp shares A rows (one table row)
  const int b_nn = tid % BN, b_kk = tid / BN;
  In a_reg[A_PER], b_reg[B_PER];

  // Persistent blocks: a shared-memory table is loaded once per block, not per tile.
  for (long long tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int tn = (int)(tile % tilesN);
    const long long rest = tile / tilesN;
    const int tm = (int)(rest % tilesM), z = (int)(rest / tilesM);
    const int m0 = tm * BM, n0 = tn * BN;
    const In* Az = static_cast<const In*>(p.A) + z * p.sAz;
    const In* Bz = static_cast<const In*>(p.B) + z * p.sBz;

    auto load = [&](int k0) {
#pragma unroll
      for (int r = 0; r < A_PER; ++r) {
        const int e = tid + r * THREADS, m = m0 + e / BK, k = k0 + e % BK;
        a_reg[r] = (m < p.M && k < p.K) ? Az[m * p.sAm + k * p.sAk] : In(0);
      }
#pragma unroll
      for (int r = 0; r < B_PER; ++r) {
        const int k = k0 + b_kk + r * B_KSTEP, n = n0 + b_nn;
        b_reg[r] = (n < p.N && k < p.K) ? Bz[k * p.sBk + n * p.sBn] : In(0);
      }
    };
    auto store = [&]() {
#pragma unroll
      for (int r = 0; r < A_PER; ++r) {
        const int e = tid + r * THREADS;
        As[e % BK * BM + e / BK] = a_reg[r];
      }
#pragma unroll
      for (int r = 0; r < B_PER; ++r) Bs[(b_kk + r * B_KSTEP) * BN + b_nn] = b_reg[r];
    };

    typename P::Acc acc[TM][TN] = {};
    load(0);
    store();
    __syncthreads();
    for (int k0 = 0; k0 < p.K; k0 += BK) {
      const bool more = k0 + BK < p.K;
      if (more) load(k0 + BK);  // global loads in flight during this tile's lookups
      auto step = [&](int kk) {
        In va[TM], vb[TN];
        read(As + kk * BM + ty * TM, va);
        read(Bs + kk * BN + tx * TN, vb);
        typename P::Op a[TM], b[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i) a[i] = P::pa(va[i], p.shift);
#pragma unroll
        for (int j = 0; j < TN; ++j) b[j] = P::pb(vb[j], p.shift);
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) P::mac(acc[i][j], a[i], b[j], t);
      };
      if (more) {
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) step(kk);
      } else {
        for (int kk = 0; kk < p.K - k0; ++kk) step(kk);  // never adds padding products
      }
      __syncthreads();
      if (more) {
        store();
        __syncthreads();
      }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int m = m0 + ty * TM + i;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int n = n0 + tx * TN + j;
        if (m < p.M && n < p.N) P::out(p.C, ((long long)z * p.M + m) * p.N + n, acc[i][j]);
      }
    }
  }
}

template <int KIND, int BM, int BN, int TM, int TN, bool SMEM>
void launch(const Params& p, size_t smem) {
  auto kernel = code_gemm_kernel<KIND, BM, BN, TM, TN, SMEM>;
  const int dev = c10::cuda::current_device();  // < MAX_DEVICES, checked by code_gemm()
  static size_t smem_set[MAX_DEVICES] = {}, per_sm_smem[MAX_DEVICES] = {};  // attributes are per device
  static int per_sm[MAX_DEVICES] = {};
  if (smem > smem_set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    smem_set[dev] = smem;
  }
  if (smem != per_sm_smem[dev]) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm[dev], kernel, THREADS, smem));
    per_sm_smem[dev] = smem;
  }
  const long long tiles = (long long)p.Z * ((p.M + BM - 1) / BM) * ((p.N + BN - 1) / BN);
  const long long cap = (long long)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * std::max(per_sm[dev], 1);
  const int grid = (int)std::min(tiles, cap);
  if (grid == 0) return;
  kernel<<<grid, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int KIND, int BM, int BN, int TM = 4, int TN = 4>
void pick(const Params& p, bool smem_ok) {
  const size_t tiles = BK * (BM + BN) * sizeof(typename Pol<KIND>::In);
  if (smem_ok && tiles + p.tab_bytes <= at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin)
    launch<KIND, BM, BN, TM, TN, true>(p, tiles + p.tab_bytes);
  else
    launch<KIND, BM, BN, TM, TN, false>(p, tiles);
}

// Tile choice, from an RTX A6000 sweep: a table read from global memory wants few
// distinct A rows per k (16x256 is up to 3.5x faster), a table in shared memory
// wants 32x128, N <= 64 wants 64x64, and fewer than SMs/2 tiles want 32x32 with 2x2
// per thread (4x the blocks).
template <int KIND>
void dispatch(const Params& p, bool smem_ok) {
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  const bool global = !smem_ok || p.tab_bytes + BK * 160 * sizeof(typename Pol<KIND>::In) > prop->sharedMemPerBlockOptin;
  const int bm = p.N <= 64 ? 64 : (p.M <= 16 || (global && p.N > 128)) ? 16 : 32;
  const int bn = p.N <= 64 ? 64 : bm == 16 ? 256 : 128;
  if ((long long)p.Z * ((p.M + bm - 1) / bm) * ((p.N + bn - 1) / bn) < prop->multiProcessorCount / 2)
    pick<KIND, 32, 32, 2, 2>(p, smem_ok);
  else if (bm == 16) pick<KIND, 16, 256>(p, smem_ok);
  else if (bm == 64) pick<KIND, 64, 64>(p, smem_ok);
  else pick<KIND, 32, 128>(p, smem_ok);
}

}  // namespace

// A [Z, M, K], B [Z, K, N] (any strides, raw 1- or 2-byte codes) -> contiguous C [Z, M, N].
at::Tensor code_gemm(const at::Tensor& A, const at::Tensor& B, const at::Tensor& T, int64_t kind,
                     int64_t shift, bool smem_ok) {
  TORCH_CHECK(kind >= 0 && kind < N_KINDS, "unknown kind");
  TORCH_CHECK(A.is_cuda() && B.device() == A.device() && T.device() == A.device(),
              "A, B and T must be on the same CUDA device");
  TORCH_CHECK(A.dim() == 3 && B.dim() == 3 && A.size(0) == B.size(0) && A.size(2) == B.size(1),
              "A [Z, M, K] and B [Z, K, N] required");
  const bool wide = kind == K_BF16 || kind == K_FP16 || kind == K_INT16;
  TORCH_CHECK(A.element_size() == (wide ? 2 : 1) && B.element_size() == A.element_size(),
              "operands must be ", wide ? 2 : 1, "-byte codes");
  const int64_t side = kind == K_F32SYM || kind == K_BF16 ? 128 : kind == K_FP16 ? 1024
                       : kind == K_INT16 ? 256 : int64_t(1) << shift;
  TORCH_CHECK(wide || kind == K_F32SYM || shift == 4 || shift == 8, "shift must be 4 or 8");
  const auto tab_type = kind == K_F32 || kind == K_F32SYM ? at::kFloat : kind == K_BF16 ? at::kByte : at::kShort;
  TORCH_CHECK(T.scalar_type() == tab_type && T.is_contiguous() && T.numel() == side * side,
              "table must be ", side, "x", side, " contiguous ", tab_type);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(T.data_ptr()) % 16 == 0, "table must be 16-byte aligned");
  TORCH_CHECK(A.get_device() < MAX_DEVICES, "device index too large");
  const int64_t Z = A.size(0), M = A.size(1), K = A.size(2), N = B.size(2);
  // Integer sums stay exact in int32 (|entry| <= 65535; INT16: |mid| <= 2 * 65535 * K).
  TORCH_CHECK(kind > K_U16 || K <= 32767, "K > 32767 could overflow the int32 sums");
  TORCH_CHECK(kind != K_INT16 || K <= 16384, "int16: K > 16384 could overflow the int32 partial sums");
  TORCH_CHECK(M < (1 << 30) && N < (1 << 30) && K < (1 << 30) && Z < INT32_MAX, "dimension too large");  // int tile math
  const auto out_type = kind == K_I16 || kind == K_U16 ? at::kInt : kind == K_INT16 ? at::kLong
                        : kind == K_BF16 ? at::kBFloat16 : kind == K_FP16 ? at::kHalf : at::kFloat;
  auto C = at::empty({Z, M, N}, A.options().dtype(out_type));
  if (C.numel() == 0) return C;
  if (K == 0) return C.zero_();
  const c10::cuda::CUDAGuard guard(A.device());
  Params p{A.data_ptr(), B.data_ptr(), T.data_ptr(), C.data_ptr(), (int)Z, (int)M, (int)N, (int)K,
           (int)shift, (int)(T.numel() * T.element_size()),
           A.stride(0), A.stride(1), A.stride(2), B.stride(0), B.stride(1), B.stride(2)};
  switch (kind) {
    case K_I16: dispatch<K_I16>(p, smem_ok); break;
    case K_U16: dispatch<K_U16>(p, smem_ok); break;
    case K_F32: dispatch<K_F32>(p, smem_ok); break;
    case K_F32SYM: dispatch<K_F32SYM>(p, smem_ok); break;
    case K_BF16: dispatch<K_BF16>(p, smem_ok); break;
    case K_FP16: dispatch<K_FP16>(p, smem_ok); break;
    default: dispatch<K_INT16>(p, smem_ok); break;
  }
  return C;
}
