"""LUT GEMM kernels, built as ``approxtorch.backend._lut_gemm``.

``lut_gemm.cu``: the int8 fast path. The half of the 256x256 int8 LUT its operands
can reach is a 64 KB int16 table in shared memory. Integer sums are exact, so results
match ``ops.gemm_int8``/``ops.bgemm_int8`` bit for bit.

``table(lut, mode)`` derives that table from a flat 65536-entry int32 CUDA LUT
(index ``(a + 128) * 256 + (b + 128)``, as ``load_lut(..., 'int8')`` returns)
and caches it on the LUT tensor until the LUT is next updated in place. It
returns None when the kernel cannot run: CPU tensors, ``APPROXTORCH_LUT_GEMM=0``,
GPUs with < 96 KB shared memory per block (Turing, Pascal), LUT entries outside
int16, or (SIGNMAG) a LUT that is not sign-symmetric. Callers then use ``ops``.

``code_gemm.cu``: ``matmul``/``bgemm`` for every precision in ``KINDS`` (see ``matmul``),
bit-identical to ``reference``. Falls back to ``reference`` where the kernel cannot run.
"""

from __future__ import annotations

import math
import os
from functools import cache

import torch

CONV, LINEAR, SIGNMAG = 0, 1, 2
MAX_IM2COL_K = 5632
SMEM_BYTES = 96 * 1024  # largest block in lut_gemm.cu: 64 KB table + tiles + MAX_IM2COL_K offsets


@cache
def _extension():
    if os.environ.get("APPROXTORCH_LUT_GEMM", "1") == "0" or not torch.cuda.is_available():
        return None
    from . import _lut_gemm

    return _lut_gemm


@cache
def _fits(device: torch.device) -> bool:
    # V100 and Ampere+ have >= 96 KB per block; Turing (T4, RTX 20xx) and Pascal fall back.
    return torch.cuda.get_device_properties(device).shared_memory_per_block_optin >= SMEM_BYTES


def table(lut: torch.Tensor, mode: int) -> torch.Tensor | None:
    """Kernel table for ``mode``: CONV (A = int8 weight, B = activation in [0, 127]),
    LINEAR (A = activation in [0, 127], B = int8 weight) or SIGNMAG (both in [-127, 127])."""
    if not lut.is_cuda or _extension() is None or not _fits(lut.device):
        return None
    version = _version(lut)
    cached = lut.__dict__.setdefault("_lut_gemm_tables", {}).get(mode)
    if version is not None and cached is not None and cached[0] == version:
        return cached[1]
    grid = lut.view(256, 256)
    if mode == CONV:
        sub = grid[:, 128:]
    elif mode == LINEAR:
        sub = grid[128:, :]
    else:
        magnitude = grid[128:, 128:]
        values = torch.arange(-127, 128, device=lut.device)
        sign = torch.where((values[:, None] < 0) ^ (values[None, :] < 0), -1, 1)
        rebuilt = sign * magnitude[values.abs()[:, None], values.abs()[None, :]]
        sub = torch.cat([magnitude, -magnitude]) if torch.equal(grid[1:, 1:], rebuilt) else None
    if sub is not None and (sub.min() < -32767 or sub.max() > 32767):
        sub = None
    result = None if sub is None else sub.to(torch.int16).contiguous().reshape(-1)
    lut._lut_gemm_tables[mode] = (version, result)
    return result


def _version(lut):
    """Cache key for tables built from ``lut``; None (never cached) for inference tensors."""
    return None if lut.is_inference() else lut._version


def _run(a, b, t, out, mode, z, m, n, k, l, sa, sb, sc, c=None, s1=None, s2=None, bias=None, vec_on_n=False):
    _extension().lut_gemm(a, b, t, out, mode, z, m, n, k, l, list(sa), list(sb), list(sc), c, s1, s2, bias, vec_on_n)
    return out


def conv(kernels, features, t, c, s1, bias=None):
    """[O, K] weights x [B, K, L] activations -> float [B, O, L] = (acc + c[o]) * s1[o] + bias[o]."""
    (o, k), (batch, _, l) = kernels.shape, features.shape
    out = torch.empty(batch, o, l, device=features.device)
    return _run(kernels, features, t, out, CONV, 1, o, batch * l, k, l,
                (0, *kernels.stride()), (0, *features.stride()), (0, *out.stride()), c, s1, None, bias)


def linear(x, weight_t, t, c, s1, s2, bias=None):
    """[M, K] activations x [K, N] weights -> float [M, N] = ((acc + c[n]) * s1[n]) * s2[n] + bias[n]."""
    (m, k), n = x.shape, weight_t.shape[1]
    out = torch.empty(m, n, device=x.device)
    return _run(x, weight_t, t, out, LINEAR, 1, m, n, k, n,
                (0, *x.stride()), (0, 0, *weight_t.stride()), (0, 0, *out.stride()), c, s1, s2, bias, True)


def batched(a, b, t):
    """Raw int32 [Z, M, N] = LUT-sum of [Z, M, K] x [Z, K, N]; any strides, SIGNMAG table."""
    (z, m, k), n = a.shape, b.shape[2]
    out = torch.empty(z, m, n, dtype=torch.int32, device=a.device)
    if out.numel() == 0 or k == 0:
        return out.zero_()
    s = b.stride()
    return _run(a, b, t, out, SIGNMAG, z, m, n, k, n,
                a.stride(), (s[0], 0, s[1], s[2]), (out.stride(0), 0, *out.stride()[1:]))


def quantize_u7(x, scale, zero, padding=(0, 0)):
    """``torch.round(x / scale + zero).clamp(0, 127).to(torch.int8)`` bit for bit in one pass,
    zero-padded by ``padding`` on the last two dims (``conv2d`` input)."""
    return _extension().quantize_u7(x, scale, zero, *padding)


def im2col_table(channels, kernel_size, dilation, height, width, device):
    """Per-k gather offsets into a padded [C, height, width] image, ``F.unfold`` order."""
    (kh, kw), (dh, dw) = kernel_size, dilation
    k = torch.arange(channels * kh * kw, device=device)
    return (k // (kh * kw) * height * width + k // kw % kh * dh * width + k % kw * dw).int()


def conv2d(x_padded, kernels, t, ktab, out_hw, stride, c, s1, bias=None):
    """Implicit-GEMM conv (no im2col buffer, K <= MAX_IM2COL_K) -> float [B, O, OH, OW]."""
    return _extension().lut_conv2d(x_padded, kernels, t, ktab, *out_hw, *stride, c, s1, bias)


# ---- every precision: code_gemm.cu -------------------------------------------------------------

KINDS = ("int4", "int8", "uint8", "int16", "fp8_e4m3", "fp8_e5m2", "fp4", "bf16", "fp16")
_I16, _U16, _F32, _F32SYM, _BF16, _FP16, _INT16 = range(7)
FP_FORMATS = {"fp8_e4m3": (4, 3, 7), "fp8_e5m2": (5, 2, 15), "fp4": (2, 1, 1)}  # exponent, fraction bits, bias
_FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def matmul(a, b, lut, kind, smem=True):
    """``a @ b`` with every product read from ``lut``; leading dims broadcast, ``a`` selects the LUT row.

    ======== ============================ ========================================== =========
    kind     operands                     lut                                        result
    ======== ============================ ========================================== =========
    int8     int8                         ``load_lut(.., 'int8')``: [(a+128)*256+b+128] int32
    int4     int8 in [-8, 7]              256 entries, [(a+8)*16 + b+8]              int32
    uint8    uint8                        ``load_lut(.., 'uint8')``: [a*256 + b]     int32
    int16    int16                        the uint8 LUT, entries in [0, 65535]       int64
    fp8_*    float8_e4m3fn / float8_e5m2  float32 [256, 256] code pair table          float32
    fp4      uint8 codes 0..15 (E2M1)     float32 [16, 16] code pair table            float32
    bf16     bfloat16                     upstream uint32 [128, 128] mantissa LUT     bfloat16
    fp16     float16                      upstream uint32 [1024, 1024] mantissa LUT   float16
    ======== ============================ ========================================== =========

    int16 is sign-magnitude over four 8x8 multipliers: with |x| = 256 h + l,
    P(a, b) = +-(T[ha, hb] << 16 + (T[ha, lb] + T[la, hb]) << 8 + T[la, lb]), sign negative iff
    exactly one operand is negative. fp8/fp4 tables come from ``fp_table``; bf16/fp16 follow the
    upstream RTL rule (``csrc/float/approx_mul_bf16.cuh``) and match ``ops.gemm_bf16_naive`` etc.
    Integer sums are exact (K <= 32767; int16 K <= 16384); float sums are float32, k ascending.
    """
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = a.expand(*batch, *a.shape[-2:]).reshape(math.prod(batch), *a.shape[-2:])
    b = b.expand(*batch, *b.shape[-2:]).reshape(math.prod(batch), *b.shape[-2:])
    out = _gemm(a, b, lut, kind, False, smem)
    return out.reshape(*batch, *out.shape[-2:])


def bgemm(x, w, lut, kind, smem=True, x_row=False):
    """Upstream bgemm layout: [B, K, L] x [O, K] -> [B, O, L], y[b, o, l] = sum_k P(w[o, k], x[b, k, l]).

    ``w`` selects the LUT row like ``matmul(w, x)`` and upstream ``bgemm_int8``/``bgemm_uint8``;
    upstream ``bgemm_bf16``/``bgemm_fp16`` index the other way round: pass ``x_row=True``.
    """
    return _gemm(w.expand(x.shape[0], *w.shape), x, lut, kind, x_row, smem)


def reference(a, b, lut, kind):
    """Torch ``matmul`` (any device): exact integer sums, float32 sums in ascending k."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = a.expand(*batch, *a.shape[-2:]).reshape(math.prod(batch), *a.shape[-2:])
    b = b.expand(*batch, *b.shape[-2:]).reshape(math.prod(batch), *b.shape[-2:])
    out = _reference(a, b, lut, kind, False)
    return out.reshape(*batch, *out.shape[-2:])


def _gemm(a, b, lut, kind, swap, smem):
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    prep = _code_table(lut, kind, swap) if a.is_cuda and _extension() is not None else None
    if prep is None:
        return _reference(a, b, lut, kind, swap)
    return _extension().code_gemm(_unit(a), _unit(b), prep[2], prep[0], prep[1], smem)


def _unit(x):
    """Coalesced loads need a unit stride on the last dim; keeps a broadcast (stride 0) batch."""
    if x.stride(2) == 1 or x.size(2) == 1:
        return x
    return x[:1].contiguous().expand(x.shape) if x.stride(0) == 0 else x.contiguous()


def _code_table(lut, kind, swap):
    """(kernel kind, shift, table) for raw operand bits, cached on ``lut``; None -> torch reference."""
    version = _version(lut)
    cache = lut.__dict__.setdefault("_code_tables", {})
    hit = cache.get((kind, swap))
    if version is not None and hit is not None and hit[0] == version:
        return hit[1]
    shift = 4 if kind in ("int4", "fp4") else 8
    if kind in ("bf16", "fp16"):
        side, mask = (128, 0xFF) if kind == "bf16" else (1024, 0x7FF)
        t = _i64(lut).view(side, side) & mask  # bit F: exponent + 1, bits F-1..0: fraction
        t = t.t() if swap else t
        prep = (_BF16, 0, t.to(torch.uint8)) if kind == "bf16" else (_FP16, 0, t.to(torch.int16))
    elif kind in ("fp8_e4m3", "fp8_e5m2", "fp4"):
        t = lut.reshape(1 << shift, 1 << shift).float()
        t = t.t() if swap else t
        flip = torch.arange(256, device=t.device) ^ 0x80
        if shift == 8 and _same(t[flip], -t) and _same(t[:, flip], -t):
            prep = (_F32SYM, 0, t[:128, :128])  # sign symmetric: 64 KB fits shared memory
        else:
            prep = (_F32, shift, t)
    else:
        t = _i64(lut).view(1 << shift, 1 << shift)
        if kind in ("int8", "int4"):  # raw two's complement bits -> LUT index: flip the sign bit
            perm = torch.arange(1 << shift, device=t.device) ^ (1 << shift - 1)
            t = t[perm][:, perm]
        t = t.t() if swap else t
        lo, hi = t.min().item(), t.max().item()
        if kind != "int16" and lo >= -0x8000 and hi <= 0x7FFF:
            prep = (_I16, shift, t.to(torch.int16))
        elif lo >= 0 and hi <= 0xFFFF:  # uint16 bits in an int16 tensor
            prep = (_INT16 if kind == "int16" else _U16, shift, (t - (t > 0x7FFF).long() * 0x10000).to(torch.int16))
        else:
            prep = None
    if prep is not None:
        prep = (prep[0], prep[1], prep[2].reshape(-1).clone())  # own aligned copy: the kernel reads 16 B
    cache[(kind, swap)] = (version, prep)
    return prep


def _i64(lut):
    return (lut.view(torch.int32) if lut.dtype == torch.uint32 else lut).long()


def _same(x, y):
    return bool(((x == y) | (x.isnan() & y.isnan())).all())


def _reference(a, b, lut, kind, swap):
    """C[z, m, n] = sum_k P(row, col) with (row, col) = (a, b), or (b, a) if ``swap``."""
    product = _products(lut, kind)
    a, b = _codes(a, kind), _codes(b, kind)
    floating = kind not in ("int4", "int8", "uint8", "int16")
    acc = torch.zeros(a.shape[0], a.shape[1], b.shape[2], device=a.device,
                      dtype=torch.float32 if floating else torch.int64)
    for k in range(a.shape[2]):
        x, y = a[:, :, k, None], b[:, None, k, :]
        acc = acc + (product(y, x) if swap else product(x, y))  # ponytail: O(K) launches, a reference
    out_type = {"int16": torch.int64, "bf16": torch.bfloat16, "fp16": torch.float16}.get(
        kind, torch.float32 if floating else torch.int32)
    return acc.to(out_type)


def _codes(x, kind):
    if kind in ("bf16", "fp16"):
        return x.view(torch.int16).long() & 0xFFFF
    return (x.view(torch.uint8) if kind.startswith("fp8") else x).long()


def _products(lut, kind):
    if kind in ("bf16", "fp16"):
        frac, dtype = (7, torch.bfloat16) if kind == "bf16" else (10, torch.float16)
        flat = _i64(lut).reshape(-1)
        return lambda x, y: _rtl(x, y, flat, 15 - frac, frac).to(torch.int16).view(dtype).float()
    if kind == "int16":
        t = _i64(lut).view(256, 256)

        def product(x, y):
            hx, lx, hy, ly = x.abs() >> 8, x.abs() & 255, y.abs() >> 8, y.abs() & 255
            magnitude = (t[hx, hy] << 16) + ((t[hx, ly] + t[lx, hy]) << 8) + t[lx, ly]
            return torch.where((x < 0) ^ (y < 0), -magnitude, magnitude)
        return product
    side, offset = {"int8": (256, 128), "int4": (16, 8), "fp4": (16, 0)}.get(kind, (256, 0))
    t = lut.reshape(side, side)
    t = t.float() if kind.startswith("fp") else _i64(t)
    return lambda x, y: t[x + offset, y + offset]


def _rtl(x, y, lut, e, f):
    """Upstream RTL product of (1 + e + f)-bit float codes as a signed code, any (e, f)."""
    fmask, emask, bias = (1 << f) - 1, (1 << e) - 1, (1 << e - 1) - 1
    entry = lut[((x & fmask) << f) | (y & fmask)] & ((2 << f) - 1)  # bit f: exponent + 1
    magnitude = ((((x >> f) & emask) + ((y >> f) & emask)) << f) - (bias << f) + entry
    code = ((x ^ y) & (1 << e + f)) | (magnitude & ((1 << e + f) - 1))  # exponent wraps
    code = torch.where(((x & ((1 << e + f) - 1)) == 0) | ((y & ((1 << e + f) - 1)) == 0), 0, code)
    return code - (code >> e + f << e + f + 1)  # two's complement view of the sign bit


def fp_table(mantissa_lut, fmt):
    """float32 code pair table for ``matmul(kind=fmt)`` from a mantissa LUT.

    ``mantissa_lut``: (2^F)^2 entries indexed ``frac_a << F | frac_b``; bit F of an entry adds one
    to the exponent, bits F-1..0 are the product fraction (as for the upstream BF16/FP16 LUTs).
    Unlike the 16-bit RTL, whose exponent wire wraps, the product keeps an exact exponent: an
    FP8/FP4 product is wider than its operands (as in MX/tensor-core MACs). Subnormal operands are
    normalized before the lookup, a zero operand gives +0, and NaN/inf codes are read as ordinary
    numbers. ``fmt``: ``FP_FORMATS`` key.
    """
    e, f, bias = FP_FORMATS[fmt]
    codes = torch.arange(1 << 1 + e + f, device=mantissa_lut.device)
    field, frac = (codes >> f) & ((1 << e) - 1), codes & ((1 << f) - 1)
    value = torch.where(field > 0, (frac + (1 << f)) * 2.0 ** (field - bias - f), frac * 2.0 ** (1 - bias - f))
    mantissa, exp = torch.frexp(value.double())  # value = mantissa * 2^exp, mantissa in [0.5, 1)
    frac = ((mantissa * 2 - 1) * (1 << f)).round().long().clamp_min(0)  # normalized fraction
    entry = _i64(mantissa_lut).reshape(-1)[frac[:, None] << f | frac[None, :]]
    product = (1 + (entry & ((1 << f) - 1)) / (1 << f)) * 2.0 ** (exp[:, None] + exp[None, :] - 2 + (entry >> f & 1))
    product = torch.where((codes[:, None] ^ codes[None, :]) >> e + f & 1 == 1, -product, product)
    return product.masked_fill((value[:, None] == 0) | (value[None, :] == 0), 0).float()


def fp4_encode(x):
    """Round to the nearest E2M1 value (ties to even, saturating at +-6) -> uint8 codes 0..15."""
    m = x.float().abs().contiguous()
    grid = torch.tensor(_FP4_GRID, device=x.device)
    mids = (grid[1:] + grid[:-1]) / 2
    down, up = torch.bucketize(m, mids), torch.bucketize(m, mids, right=True)
    code = torch.where(up % 2 == 0, up, down)  # differ only on a tie: take the even code
    return (code + 8 * x.signbit()).to(torch.uint8)
