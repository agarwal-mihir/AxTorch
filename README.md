<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/axtorch-logo-dark.svg">
    <img alt="AxTorch" src="assets/axtorch-logo.svg" width="640">
  </picture>
</p>

# AxTorch

> **AxTorch is a fork of [ApproxTorch](https://github.com/woshimark666/ApproxTorch)
> by Ke Ma (MIT).** It adds transformer/LLM support and a faster LUT-GEMM CUDA
> kernel and keeps everything else unchanged. The Python package is still
> `approxtorch`, so `import approxtorch as at` keeps working as a drop-in
> replacement. All credit for the original library belongs to its authors; if you
> use AxTorch, please [cite ApproxTorch](#citation).

## What AxTorch adds

### Transformers and LLMs (`approxtorch.transformer`)

`approximate_model` replaces each `nn.Linear` in a model with a LUT-approximate
`LutLinear`. It also routes both attention matmuls (Q·Kᵀ and softmax·V) of any
Hugging Face model through the LUT. Since pretrained models have no QAT
checkpoint, quantization is symmetric int8 by default (other formats:
`kind=`, [below](#other-precisions-lut_kernelmatmul)): weights are scaled per output
channel once, activations per token, and attention operands per row or column.
Elementwise products such as RoPE, norms and GLU gates stay exact.

```python
import torch
import approxtorch as at
from transformers import AutoModelForCausalLM

v = torch.arange(-128, 128)
lut = (v[:, None] * v[None, :]).int().reshape(-1).cuda()  # exact; or at.load_lut.load_lut('my_mult.txt', 'int8')
model = AutoModelForCausalLM.from_pretrained('google/gemma-3-270m').cuda().eval()
replaced = at.transformer.approximate_model(model, lut)  # skips lm_head by default
# lut.copy_(other_lut) switches every patched layer to another multiplier in place
```

Lower-level building blocks:
- `approx_matmul(a, b, lut)`: float a @ b through the LUT.
- `approx_attention(q, k, v, lut, attn_mask)`: attention for plain PyTorch modules.
- `lut_matmul`: the integer version of `approx_matmul`.

`nn.MultiheadAttention` is not patched.

Gemma 3 270M on WikiText-2 (8,192 tokens, RTX A6000), with approximate
multipliers from an NSGA-II search:

| Design | MRED | Perplexity | Next-token acc. |
|---|---|---|---|
| float32 | — | 26.09 | 39.6% |
| exact int8 | 0 | 27.01 | 39.2% |
| approximate, MRED 0.030 | 0.030 | 30.73 | 38.1% |
| approximate, MRED 0.046 | 0.046 | 37.55 | 35.3% |
| approximate, MRED 0.081 | 0.081 | 70.30 | 28.7% |
| approximate, MRED 0.117 | 0.117 | 365.55 | 15.8% |

A ViT-Small trained on MNIST stays between 97.7% and 98.1% top-1 across the same
designs.

### LUT-GEMM kernel (`approxtorch.backend.lut_kernel`)

`lut_gemm.cu` is built as `approxtorch.backend._lut_gemm` and replaces
`gemm_int8`, `bgemm_int8` and `im2col_int8` on CUDA. Its results are
**bit-identical** to those ops.
- It keeps the half of the 256×256 LUT that the operands can reach in shared
  memory, as a 64 KB int16 table.
- Convolutions gather their inputs directly (implicit GEMM, no im2col buffer).
- Activations are quantized and zero-padded in a single pass.
- The zero-point correction, scaling and bias are fused into the GEMM.

The transformer path uses it automatically for every matmul, as one batched
launch over all (sample, head) pairs. For CNN layers, the raw ops are exposed:
`table`, `conv`, `conv2d`, `linear`, `batched` and `quantize_u7`. The layers
from `convert_int_model` do not use the kernel yet.

Op-level timings on an RTX A6000, using a random LUT:

| Op | ApproxTorch ops | LUT-GEMM | Speed-up |
|---|---:|---:|---:|
| 3×3 conv, 64→64 ch, 8×8, batch 256 (`im2col_int8` + `bgemm_int8` vs `conv2d`) | 2.39 ms | 0.50 ms | 4.8× |
| Linear 2048×640 @ 640×2048 (`gemm_int8` vs `linear`) | 20.8 ms | 2.0 ms | 10.4× |
| Attention Q·Kᵀ, 1536 (sample, head) pairs of 65×64 @ 64×65 (loop of `gemm_int8` vs one `batched` launch) | 123 ms | 0.91 ms | 136× |

These baselines are the default ApproxTorch ops. ApproxTorch also ships faster
float-LUT variants (`gemm_fake_int8_claude`, `bgemm_fake_int8_claude`, …). Against
those, LUT-GEMM is about 1.5–3× faster for convolutions, 0.8–1.8× for linears and
about 97× for batched attention.

Per-candidate evaluation on an RTX A6000, measured in a downstream search
framework that wires these ops into its integer layers (full test set,
batch 256):

| Workload | ApproxTorch ops | LUT-GEMM | Speed-up |
|---|---:|---:|---:|
| LeNet-5, MNIST | 0.058 s | 0.016 s | 3.6× |
| MLP, MNIST | 0.027 s | 0.012 s | 2.3× |
| ResNet-20, CIFAR-10 | 1.774 s | 0.391 s | 4.5× |
| ResNet-50, CIFAR-10 | 43.6 s | 9.93 s | 4.4× |
| ViT-S MNIST, 3 designs | 205 s | 30 s | 6.8× |

The kernel needs at least 96 KB of shared memory per block. That covers V100,
A100, H100 and RTX 30/40/A-series GPUs. `table()` returns `None` and callers fall
back to the original ops in these cases:
- Turing (T4, RTX 20xx) and older GPUs;
- LUT entries outside the int16 range;
- convolutions with C·kh·kw > 5632.

Set `APPROXTORCH_LUT_GEMM=0` to force the fallback.

### Other precisions (`lut_kernel.matmul`)

`matmul(a, b, lut, kind)` sends every product of `a @ b` through the LUT for
nine operand formats, using one templated kernel (`code_gemm.cu`). `a` selects
the LUT row and leading dims broadcast. `bgemm(x, w, lut, kind)` takes the
upstream `[B, K, L] × [O, K]` layout. `reference(...)` is a pure-torch version of
`matmul` that runs on any device and also serves as the CPU fallback.

| `kind` | Operands | LUT | Result |
|---|---|---|---|
| `int4` | int8 in [-8, 7] | 256 int32, index `(a+8)*16 + b+8` | int32 |
| `int8` | int8 | `load_lut(.., 'int8')`, any LUT (not only sign-symmetric) | int32 |
| `uint8` | uint8 | `load_lut(.., 'uint8')` | int32 |
| `int16` | int16 | the uint8 LUT, entries in [0, 65535] | int64 |
| `fp8_e4m3`, `fp8_e5m2` | `torch.float8_e4m3fn` / `float8_e5m2` | float32 256×256 code-pair table from `fp_table` | float32 |
| `fp4` | uint8 E2M1 codes from `fp4_encode` | float32 16×16 code-pair table from `fp_table` | float32 |
| `bf16` | bfloat16 | upstream 128×128 mantissa LUT | bfloat16 |
| `fp16` | float16 | upstream 1024×1024 mantissa LUT | float16 |

Semantics:
- **Bit-identity.** int8/uint8 match `gemm_int8`/`bgemm_int8`/`gemm_uint8`/`bgemm_uint8`
  bit for bit. bf16/fp16 match `gemm_bf16_naive` etc., including the upstream
  RTL's 5/8-bit exponent wrap. `ops.gemm_bf16`, `gemm_fp16`, `bgemm_bf16` and
  `bgemm_fp16` now run on this kernel. Integer sums are exact. Float sums are
  float32, accumulated in ascending k.
- **int16** is sign-magnitude over four 8×8 multipliers. With |x| = 256h + l, the
  product is ±(T[ha,hb]·2¹⁶ + (T[ha,lb] + T[la,hb])·2⁸ + T[la,lb]).
- **FP8/FP4 tables.** `fp_table(mantissa_lut, fmt)` builds the code-pair table
  from an upstream-style mantissa LUT. Unlike the 16-bit RTL, it keeps the
  product's exponent exact, because an FP8/FP4 product is wider than its
  operands (as in MX/tensor-core MACs). Subnormals are normalized before the
  lookup, and a zero operand gives +0. Any other float32 table works too.
- **LUT row in `bgemm`.** By default `w` selects the LUT row, as in upstream
  `bgemm_int8`/`bgemm_uint8`. Upstream `bgemm_bf16`/`bgemm_fp16` index the other
  way: pass `x_row=True`.
- **Shared memory.** Tables that fit in shared memory (int4, fp4, bf16, and
  sign-symmetric fp8 at 64 KB) are kept there. The rest (int8/uint8/int16 at
  128 KB, asymmetric fp8 at 256 KB, fp16 at 2 MB) are read through L1. On
  A100/H100, 128 KB tables would fit in shared memory, but this is untested.

The transformer API takes the same `kind` (`approximate_model(model, lut,
kind='fp8_e4m3')`). Quantization is symmetric, scaled to the format's max:
- bf16 is a plain cast;
- fp16 is scaled to |x| ≤ min(16, √(2¹³/K)) for a K-long reduction, with
  |x| < 2⁻⁷ flushed to zero. Products then never reach the RTL's exponent wrap,
  and sums stay inside the fp16 range;
- uint8 needs zero points and is not offered.

int8 with a sign-symmetric LUT keeps using the faster int8 kernel above.

Max relative error vs float32 of a random linear layer plus attention, with
exact multipliers (tests use about 2× these as tolerances):

| int4 | int8 | int16 | fp8_e4m3 | fp8_e5m2 | fp4 | bf16 | fp16 |
|---|---|---|---|---|---|---|---|
| 0.20 | 0.011 | 4e-5 | 0.080 | 0.11 | 0.19 | 0.007 | 6e-4 |

Timings in ms on an RTX A6000 (a shared GPU, best of 3 runs). In parentheses:
the fastest upstream op (`gemm_fake_int8_claude`, `gemm_fake_uint8_claude`,
`gemm_bf16`, `gemm_fp16`):

| `kind` | 2048×640 @ 640×2048 | 512×512 @ 512×512 | 64×4096 @ 4096×4096 | 256×784 @ 784×300 | 8×4096 @ 4096×4096 |
|---|---:|---:|---:|---:|---:|
| int4 | 1.50 | 0.10 | 0.76 | 0.07 | 0.67 |
| int8 | 2.06 (2.83) | 0.18 (0.21) | 1.18 (1.30) | 0.17 (0.13) | 0.70 (0.43) |
| uint8 | 2.05 (2.99) | 0.18 (0.21) | 1.22 (1.30) | 0.17 (0.13) | 0.70 (0.43) |
| int16 | 10.5 | 0.64 | 5.06 | 0.46 | 2.56 |
| fp8 (either) | 1.94 | 0.13 | 1.02 | 0.10 | 0.78 |
| fp4 | 1.61 | 0.11 | 0.80 | 0.07 | 0.69 |
| bf16 | 4.66 (11.0) | 0.20 (0.35) | 2.32 (4.09) | 0.12 (0.19) | 1.46 (0.51) |
| fp16 | 23.1 (49.9) | 0.50 (1.29) | 3.87 (12.6) | 0.71 (0.68) | 2.36 (0.97) |
| fp32 `torch.mm` | 0.27 | 0.025 | 0.12 | 0.02 | 0.12 |

`bgemm` with X of shape 256×576×64 and W of 64×576: bf16 takes 1.26 ms (1.65
upstream) and fp16 6.85 ms (6.21).

Float sums must stay in k order for bit-identity, so K is never split across
blocks. As a result, very skinny products (M ≤ 16, e.g. LLM decode) run 1.5–3×
slower than upstream.

Tests (`python -m unittest discover -s tests`) check the kernels bit for bit
against `gemm_int8`/`bgemm_int8`/`im2col_int8`, `gemm_uint8`/`bgemm_uint8`,
`gemm_bf16_naive`/`gemm_fp16_naive` and the torch reference for every kind. They
also check tiny Gemma 3 and ViT models against eager attention with an exact LUT.

---

The rest of this README is the original ApproxTorch documentation.

**ApproxTorch** is a PyTorch extension for simulating **approximate multipliers** inside Convolutional Neural Networks — for both **training (QAT)** and **inference** — entirely on the GPU.

Approximate multipliers are hardware circuits that trade a small amount of arithmetic accuracy for large savings in power and area. Before taping out such hardware, you need to know how much CNN accuracy you lose — and how much you can recover by retraining. ApproxTorch answers this question by replacing every multiplication inside convolution layers with a **look-up table (LUT)** of your multiplier's actual behavior, computed by highly optimized custom CUDA kernels.

Because the simulation is purely LUT-driven, **any 8-bit (signed or unsigned)
approximate multiplier can be simulated**. ApproxTorch also supports normalized-
mantissa LUTs for approximate **FP16 and BF16** GEMM and BGEMM.

```text
 FP32 input ──► quantize (int8) ──► im2col ──► LUT-based approximate GEMM ──► dequantize ──► FP32 output
                                                  (custom CUDA kernel)
```

## Features

- 🚀 **GPU-accelerated**: hand-written CUDA kernels for LUT-based approximate batched GEMM — fast enough to retrain networks like ResNet on ImageNet-scale data.
- 🎯 **Train *and* infer**: full autograd support, so you can do approximate-multiplier-aware retraining (QAT), not just evaluation.
- 🧮 **Any 8-bit multiplier**: behavior is defined entirely by a 256×256 LUT text file.
- 🧠 **Approximate FP16/BF16**: LUT mantissa GEMM/BGEMM with ordered FP32
  accumulation and one final 16-bit conversion.
- 🔁 **Gradient estimators** for backpropagating through the non-differentiable LUT:
  - **STE** — straight-through estimator (default)
  - **LRE** — linear-regression-estimated gradient LUTs
  - **Custom** — one derivative for every quantized `(x, w)` pair
- ⚖️ **EMA quantization**: per-tensor activation scale and per-channel weight scale are EMA-updated during training, with arbitrary weight bit-width (**3–8 bit**) for int8 weights.
- 🔌 **Separated model conversion**: `at.convert_int_model(...)` handles integer Conv2d; `at.convert_float_model(...)` handles FP16/BF16 Conv2d and Linear.
- 🖥️ **Multi-GPU (DDP) support**: activation scales are synchronized across ranks automatically.

## Requirements

- Linux (Windows is untested)
- Python ≥ 3.10
- PyTorch ≥ 2.4 built with CUDA ≥ 11.8
- A CUDA-capable GPU (kernels are compiled with `-arch=native` for your local GPU)
- `ninja` is recommended for faster compilation (`pip install ninja`)

## Installation

```bash
git clone https://github.com/agarwal-mihir/AxTorch.git
cd AxTorch
pip uninstall -y approxtorch   # AxTorch installs the same `approxtorch` package
pip install . --no-build-isolation
```

The distribution is named `axtorch` but still provides the `approxtorch` package,
so uninstall any existing ApproxTorch first. `approxtorch.transformer` needs
`transformers` (tested with 5.19) only for `approximate_model(..., attention=True)`.
The original library lives at https://github.com/woshimark666/ApproxTorch.

Or, for an editable install with [uv](https://github.com/astral-sh/uv):

```bash
./install.sh        # runs: uv pip install -e . --no-build-isolation
```

The CUDA extension is compiled during installation, which takes a few minutes.

## Quick Start

### 1. Prepare the multiplier LUT

Describe your approximate multiplier as a plain-text file containing a **256×256 matrix** of integers: entry *(i, j)* is the output of your multiplier for inputs `A = i − 128` and `B = j − 128` (signed int8, row-major, indices not included in the file):

```text
16384  16256  16128  ...        # A = -128:  (-128)×(-128), (-128)×(-127), ...
16256  16129  ...               # A = -127
  ...
                     ...  16129 # A =  127
```

An exact-multiplier example is provided at `test/exact_int8.txt`; LUTs of several published approximate multipliers are in `test/` as well (e.g. `mul8u_syn1.txt`, `venka.txt`, `zhang.txt`).

### 2. Convert your model and train

```python
import torch
import approxtorch as at
from torchvision.models import resnet18

device = torch.device('cuda')

# 1. load the LUT and convert it to the float32 format required by CUDA kernels
lut = at.load_lut.load_lut('test/exact_int8.txt', qtype='int8').to(
    device=device, dtype=torch.float32)

# 2. take any FP32 model and replace its Conv2d layers with approximate ones
model = resnet18(num_classes=10)
model = at.convert_int_model(
    model, lut,
    qtype='int8',             # 'int8' | 'uint8'
    grad='ste',               # gradient estimator: 'ste' | 'lre' | 'custom'
    ignore_first_conv=True,   # keep the first conv exact (common QAT practice)
    weight_bits=8,            # 3–8 for int8; uint8 is fixed at 8
).to(device)

# 3. business as usual — training and inference work like any PyTorch model
out = model(torch.randn(8, 3, 32, 32, device=device))
out.sum().backward()
```

During training, both activation and per-channel weight scales are updated as exponential moving averages (controlled by `scale_momentum`). Call `model.eval()` (or `freeze_scale()` on a layer) to stop scale updates for inference.

## Gradient Estimators

An approximate multiplier is a discrete function, so its true gradient is not defined. ApproxTorch provides three ways to backpropagate through it:

| `grad=`   | Idea | Extra inputs needed |
|-----------|------|---------------------|
| `'ste'`   | Pretend the multiply was exact (straight-through estimator) | none |
| `'lre'`   | Fit a line to each row/column of the LUT; use slopes as gradients | `dx`, `dw` gradient LUTs |
| `'custom'` | Use one derivative for each quantized `(x, w)` pair | 256×256 `dx`, `dw` gradient LUTs |

The helper module `approxtorch.grad_lut` generates these extra inputs directly from your multiplier LUT:

```python
import approxtorch as at

# LRE: per-row / per-column linear regression slopes
grad_a, grad_b = at.grad_lut.lre('my_multiplier.txt', qtype='int8', save_path='my_mult')
dx, dw = at.load_lut.load_lre_grad_lut('my_mult_lre_grad_a.txt', 'my_mult_lre_grad_b.txt')
model = at.convert_int_model(model, lut, grad='lre', dx=dx.cuda(), dw=dw.cuda())
```

For a pair-wise custom gradient, `dx[x + 128, w + 128]` supplies the
derivative with respect to `x`, while `dw[x + 128, w + 128]` supplies the
derivative with respect to `w`:

```python
dx, dw = at.load_lut.load_custom_grad_lut('custom_dx.txt', 'custom_dw.txt')
model = at.convert_int_model(
    model, lut, grad='custom',
    dx=dx.cuda(), dw=dw.cuda(),
)
```

A smoothing + central-difference method (`at.grad_lut.DATE`) is also included for research comparison.

### BF16 mantissa gradient LUTs

BF16 layers support a custom CUDA backward using two precomputed, fixed FP32
gradient LUTs. Each tensor must be contiguous and one-dimensional with shape
`[16384]`; both stay on the same CUDA device as the operands. Prepare and upload
them once before training:

```python
device = torch.device("cuda")
grad_x_lut = (
    torch.load("bf16_grad_x.pt", map_location="cpu", weights_only=True)
    .detach().float().reshape(-1).contiguous().to(device)
)
grad_w_lut = (
    torch.load("bf16_grad_w.pt", map_location="cpu", weights_only=True)
    .detach().float().reshape(-1).contiguous().to(device)
)
lutb = at.load_lut.load_float_lut("my_bf16_lut.pt", qtype="bf16").to(device)
model = at.convert_float_model(
    model, lutb, qtype="bf16", ignore_first_conv=False,
    grad_x_lut=grad_x_lut, grad_w_lut=grad_w_lut,
).to(device=device, dtype=torch.bfloat16)
```

Supplying both gradient LUTs selects custom gradients in BF16 Conv2d and Linear.
Omitting both retains the existing STE. The same keyword arguments are accepted
by `Conv2d_bf16`, `Linear_bf16`, `conv2d_bf16`, and `linear_bf16`; FP16 keeps its
existing interface. Modules register the tables as buffers, preserve their FP32
dtype when casting the model, and share already resident tensors. The forward
mantissa LUT remains a separate uint32 tensor with shape `[128, 128]`, and the
forward algorithm is unchanged.

The flattened gradient index always keeps the input fraction first:

```cpp
index = ((bits_x & 0x7F) << 7) | (bits_w & 0x7F);
```

For normal BF16 operands, table entries are `df/dm_x` and `df/dm_w` for the
approximate mantissa product `f(m_x, m_w)`. CUDA propagates

\[
g_x = g_{out}\,s_w\,2^{E_w-127}\,\mathrm{grad\_x\_lut}[index],\qquad
g_w = g_{out}\,s_x\,2^{E_x-127}\,\mathrm{grad\_w\_lut}[index].
\]

Intermediate multiplication and reduction use FP32; returned gradients keep
the BF16 operand shapes and dtype. GEMM and BGEMM sum over the existing
contraction dimensions, including all batch and spatial positions for shared
weights. Kernels only read the supplied gradient tables and do not generate or
update them. This custom backward supports first-order gradients only.

Signed-zero operands still select fraction zero in the gradient LUTs:
when `x` is zero and `w` is nonzero, `g_x` uses `grad_x_lut[0, F_w]` with
`w`'s sign and exponent scale; when `w` is zero and `x` is nonzero, `g_w`
uses `grad_w_lut[F_x, 0]` with `x`'s sign and exponent scale (indices here
show the logical 128x128 layout). Only a zero **opposite** operand masks a
partial: `w == 0` makes `g_x = 0`, and `x == 0` makes `g_w = 0`, before
multiplying by the upstream gradient. Both partials are zero when both
operands are zero. This extends the LUT gradient rule to zero; it does not
fall back to the exact-product derivative. Forward signed-zero handling is
unchanged.

Nonzero operands with exponent fields 0 or 255 retain their raw fraction and
use `E - 127`; no IEEE special-value propagation or flush-to-zero rule is
added. The forward's eight-bit exponent wraparound remains unchanged. The
exact-product gradient identity is tested for normal and zero operands using
`grad_x_lut[index] = 1 + F_w/128` and `grad_w_lut[index] = 1 + F_x/128`.

Run the correctness tests after building the extension:

```bash
python -m pytest approxtorch/backend/csrc/float/test_bf16_custom_grad.py -q
```

## API Overview

### Model conversion

```python
at.convert_int_model(
    model,                    # any nn.Module
    lut,                      # LUT tensor from at.load_lut.load_lut(...)
    qtype='int8',             # 'int8' | 'uint8'
    grad='ste',               # 'ste' | 'lre' | 'custom'
    dx=None, dw=None,         # gradient LUTs, required for 'lre'/'custom'
    ignore_first_conv=True,   # leave the first Conv2d untouched
    scale_momentum=0.05,      # EMA momentum for quantization statistics
    update_scale=True,        # update statistics while training
    weight_bits=8,            # 3–8 for int8; uint8 is fixed at 8
)
```

Floating-point conversion has its own API and converts both Conv2d and Linear:

```python
# Generate these files once. Both FP16 and BF16 LUT tensors are uint32.
# python tools/generate_float_mantissa_lut.py --kind all --format all \
#     --output-dir /tmp/approxtorch-luts
lut16 = at.float_lut.load_exact_lut(
    "fp16", "cuda:0", directory="/tmp/approxtorch-luts"
)
model = at.convert_float_model(
    model,
    lut16,
    qtype="fp16",             # "fp16" | "bf16"
    ignore_first_conv=True,   # affects Conv2d only; every Linear is converted
    optimized=True,
).to(device="cuda", dtype=torch.float16)
```

For arbitrary approximate mantissa LUT files, the general loader accepts both
short and long dtype names and returns a contiguous CPU uint32 tensor:

```python
lut16 = at.load_lut.load_lut("my_fp16_lut.bin", qtype="float16").cuda()
lutb = at.load_lut.load_float_lut("my_bf16_lut.pt", qtype="bf16").cuda()
```

Supported inputs are whitespace-delimited text, raw little-endian uint32
`.bin`, and PyTorch `.pt`/`.pth` tensors. Use `float_lut.load_exact_lut` when
loading generated exact LUTs with manifest/hash verification.

The converted Conv2d and Linear weights and biases are nn.Parameter tensors in
torch.float16 or torch.bfloat16, respectively. The converter does not cast
unreplaced layers automatically, so cast the surrounding model and its inputs
to a compatible dtype as shown above.

### Layers (`approxtorch.nn`)

Approximate layers can also be used directly, just like their `torch.nn` counterparts:

- `Conv2d_int8` — signed 8-bit approximate convolution
- `Conv2d_uint8` — unsigned 8-bit approximate convolution
- `Conv2d_fp16` / `Conv2d_bf16` — fixed-dtype approximate convolution
- `Linear_fp16` / `Linear_bf16` — fixed-dtype approximate linear layer

### Calibration (`approxtorch.calib`)

For static quantization, collect activation/weight scales on a calibration set before converting:

```python
at.calib.calibrate_int8(model, train_loader, num_pictures=1024,
                        save_path='model_calibrated.pth', weight_bits=8)
```

Calibration writes the activation scale and the EMA weight scale into module
buffers, so they can be loaded after model conversion.

### Low-level ops (`approxtorch.backend.ops`)

If you want to build your own layers, the raw CUDA ops are exposed as PyTorch
custom ops, including the integer operations and `approx_mul_fp16`,
`approx_mul_bf16`, `gemm_fp16`, `gemm_bf16`, `bgemm_fp16`, and `bgemm_bf16`
(plus their `_naive` reference kernels).

The matching STE autograd wrappers are split by dtype:

```python
from approxtorch.nn import bgemm_fp16, bgemm_bf16

y16 = bgemm_fp16.bgemm_fp16_ste(x16, w16, lut16)
yb = bgemm_bf16.bgemm_bf16_ste(xb, wb, lutb)
```

BF16 custom-gradient wrappers use the same forward operators and add the two
flat FP32 gradient tables:

```python
from approxtorch.nn import (
    approx_mul_bf16_custom, gemm_bf16_custom, bgemm_bf16_custom,
)

# Elementwise: x and w have the same shape.
y = approx_mul_bf16_custom(x, w, lutb, grad_x_lut, grad_w_lut)
# GEMM: A [M,K], B [K,O] -> [M,O].
y = gemm_bf16_custom(A, B, lutb, grad_x_lut, grad_w_lut)
# BGEMM: X [N,K,L], W [O,K] -> [N,O,L].
y = bgemm_bf16_custom(X, W, lutb, grad_x_lut, grad_w_lut)
```

## Repository Layout

```text
approxtorch/
├── backend/            # custom op registration + CUDA/C++ kernels
│   ├── lut_gemm.cu     # AxTorch: shared-memory LUT-GEMM kernel (_lut_gemm)
│   ├── lut_kernel.py   # AxTorch: Python wrappers and fallback checks
│   └── csrc/
│       ├── cuda/       # approximate (b)gemm, im2col, LUT lookup, backward kernels
│       └── cpu/        # CPU reference implementations
├── nn/                 # approximate integer and FP16/BF16 Conv2d/Linear layers
├── convert_model.py    # split integer and FP16/BF16 model conversion
├── load_lut.py         # LUT and gradient-LUT loaders
├── grad_lut.py         # gradient LUT generation (LRE, DATE)
├── quant_utils.py      # calibration utilities
└── calib.py            # standalone min-max calibration helper
└── transformer.py      # AxTorch: LutLinear, LUT attention, approximate_model
assets/                 # AxTorch logo
test/                   # example LUTs, ResNet-20/CIFAR-10 scripts, kernel tests
tests/                  # AxTorch: unittest suite for the kernel and transformer path
```

## Citation

AxTorch is built on ApproxTorch. If you use either in your work, please cite:

```bibtex
@INPROCEEDINGS{10031519,
  author={Ma, Ke and Kimura, Shinji},
  booktitle={2022 19th International SoC Design Conference (ISOCC)},
  title={ApproxTorch: An Approximate Multiplier Evaluation Environment for CNNs based on Pytorch},
  year={2022},
  pages={77-78},
  doi={10.1109/ISOCC56007.2022.10031519}}
```

## License

[MIT](LICENSE). The original ApproxTorch copyright (Ke Ma) is retained. The AxTorch
additions are also MIT.

## Famous quote

[![Readme Quotes](https://quotes-github-readme.vercel.app/api?type=horizontal&theme=dark&quote=喝可乐不加冰等于没喝)](https://github.com/piyushsuthar/github-readme-quotes)
