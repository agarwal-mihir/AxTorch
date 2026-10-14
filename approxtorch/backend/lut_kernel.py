"""Shared-memory LUT GEMM (``lut_gemm.cu``, built as ``approxtorch.backend._lut_gemm``).

The 256x256 int8 LUT keeps the half its operands can reach as a 64 KB int16
table in shared memory. Integer sums are exact, so results match
``ops.gemm_int8``/``ops.bgemm_int8`` bit for bit (2.3-6.8x faster end to end, see README).

``table(lut, mode)`` derives that table from a flat 65536-entry int32 CUDA LUT
(index ``(a + 128) * 256 + (b + 128)``, as ``load_lut(..., 'int8')`` returns)
and caches it on the LUT tensor until the LUT is next updated in place. It
returns None when the kernel cannot run: CPU tensors, ``APPROXTORCH_LUT_GEMM=0``,
GPUs with < 96 KB shared memory per block (Turing, Pascal), LUT entries outside
int16, or (SIGNMAG) a LUT that is not sign-symmetric. Callers then use ``ops``.
"""

from __future__ import annotations

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
    cached = lut.__dict__.setdefault("_lut_gemm_tables", {}).get(mode)
    if cached is not None and cached[0] == lut._version:
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
    lut._lut_gemm_tables[mode] = (lut._version, result)
    return result


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
