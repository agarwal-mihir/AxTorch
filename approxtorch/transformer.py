"""LUT-approximate matmuls for transformers: Linear layers and attention.

Every multiply in a GEMM goes through the multiplier LUT, with ``a`` the
activation (row) operand. ``kind`` picks the precision and the LUT format
(``lut_kernel.matmul``); the default int8 takes the flat 65536-entry int32
tensor from ``load_lut(path, 'int8')``, indexed ``(a + 128) * 256 + (b + 128)``.
Pretrained models have no QAT checkpoint, so quantization is symmetric (scaled
to the format's max): weights per output channel (once), activations per token
(every call), and both attention operands per row/column. bf16 is a plain
cast; fp16 is scaled to |x| <= t = min(16, sqrt(2^13 / K)) for a K-long reduction,
with |x| < 2^-7 flushed to zero: products stay in [2^-14, 256], inside the 5-bit
exponent of the upstream RTL (which wraps), and K * 4t^2 <= 2^15 keeps the fp16
sum finite even for approximate products (< 4x exact). uint8 needs
zero points and is not offered here. Elementwise products (RoPE, norms, GLU
gates) stay exact.

On CUDA, int8 with a sign-symmetric LUT runs on the shared-memory int8 kernel
and everything else on ``lut_kernel.matmul``, one batched launch per matmul.
"""

from __future__ import annotations

import math
from fnmatch import fnmatch
from functools import partial

import torch
from torch import nn

from .backend import lut_kernel


_INT_MAX = {"int8": 127, "int4": 7, "int16": 32767}
_FLOAT8 = {"fp8_e4m3": (torch.float8_e4m3fn, 448.0), "fp8_e5m2": (torch.float8_e5m2, 57344.0)}


def quantize(x: torch.Tensor, dim: int, kind: str = "int8") -> tuple[torch.Tensor, torch.Tensor]:
    """Operand codes for ``kind`` and the scale (x ~ codes * scale), symmetric per ``dim`` slice.

    ``dim`` is the matmul's reduction dim: the fp16 range depends on its length."""
    if kind == "bf16":
        return x.to(torch.bfloat16), x.new_ones(())
    amax = x.abs().amax(dim, keepdim=True).clamp_min(1e-12)
    if kind in _INT_MAX:
        top = _INT_MAX[kind]
        scale = amax / top
        dtype = torch.int16 if kind == "int16" else torch.int8
        return torch.round(x / scale).clamp(-top, top).to(dtype), scale
    if kind in _FLOAT8:
        dtype, top = _FLOAT8[kind]
        scale = amax / top
        return (x / scale).clamp(-top, top).to(dtype), scale  # e4m3fn has no inf: > 448 would be NaN
    if kind == "fp4":
        scale = amax / 6
        return lut_kernel.fp4_encode(x / scale), scale
    if kind == "fp16":  # no RTL exponent wrap, no fp16 overflow of the sum (approximate products < 4ab)
        scale = amax / min(16.0, math.sqrt(2.0 ** 13 / max(x.shape[dim], 1)))
        y = x / scale
        return y.masked_fill(y.abs() < 2 ** -7, 0).to(torch.float16), scale
    raise ValueError(f"kind must be one of {(*_INT_MAX, *_FLOAT8, 'fp4', 'bf16', 'fp16')}, got {kind!r}")


def lut_matmul(a, b, lut, symmetric=False, kind="int8") -> torch.Tensor:
    """``a @ b`` of ``kind`` codes through the LUT as float32, batched over broadcast leading dims.

    ``symmetric`` (int8): both operands are in [-127, 127] (``quantize`` output), so a
    sign-symmetric CUDA LUT can run on the shared-memory int8 kernel (bit-identical).
    """
    t = lut_kernel.table(lut, lut_kernel.SIGNMAG) if symmetric and kind == "int8" else None
    if t is None:
        return lut_kernel.matmul(a, b, lut, kind).float()
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = a.expand(*batch, *a.shape[-2:]).reshape(math.prod(batch), *a.shape[-2:])
    b = b.expand(*batch, *b.shape[-2:]).reshape(math.prod(batch), *b.shape[-2:])
    out = lut_kernel.batched(a, b, t).float()
    return out.reshape(*batch, *out.shape[-2:])


def approx_matmul(a, b, lut, kind="int8") -> torch.Tensor:
    """Float ``a @ b`` with a quantized per row and b per column."""
    a_q, a_scale = quantize(a.float(), -1, kind)
    b_q, b_scale = quantize(b.float(), -2, kind)
    return lut_matmul(a_q, b_q, lut, True, kind) * a_scale * b_scale


class LutLinear(nn.Module):
    def __init__(self, linear: nn.Linear, lut: torch.Tensor, kind: str = "int8") -> None:
        super().__init__()
        weight_q, weight_scale = quantize(linear.weight.detach().float().t(), 0, kind)
        weight_q = weight_q.contiguous()
        self.code_dtype = weight_q.dtype
        if weight_q.is_floating_point():  # raw ints (fp8, bf16, fp16), so model.half()/.to(dtype) skip them
            weight_q = weight_q.view({1: torch.uint8, 2: torch.int16}[weight_q.element_size()])
        self.register_buffer("weight_q", weight_q)
        # float32 bits: an int16/e5m2 scale (~1e-6) would be subnormal after model.half()
        self.register_buffer("weight_scale", weight_scale.view(torch.int32))
        self.register_buffer(
            "bias", None if linear.bias is None else linear.bias.detach().float()
        )
        self.lut, self.kind = lut, kind

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        scale = state_dict.get(prefix + "weight_scale")
        if scale is not None and scale.is_floating_point():  # AxTorch <= 2.0 stored the float scale
            state_dict[prefix + "weight_scale"] = scale.float().view(torch.int32)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q, x_scale = quantize(x.float().reshape(-1, x.shape[-1]), -1, self.kind)
        weight_q = self.weight_q.view(self.code_dtype)
        weight_scale = self.weight_scale.view(torch.float32)
        y = lut_matmul(x_q, weight_q, self.lut, True, self.kind) * x_scale * weight_scale
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], -1).to(x.dtype)


def approx_attention(query, key, value, lut, attn_mask=None, scale=None, softcap=None, kind="int8"):
    """softmax(Q K^T * scale [+ additive mask]) V with both matmuls through the LUT."""
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    scores = approx_matmul(query, key.transpose(-2, -1), lut, kind) * scale
    if softcap is not None:
        scores = torch.tanh(scores / softcap) * softcap
    if attn_mask is not None:
        scores = scores + attn_mask
    return approx_matmul(scores.softmax(-1), value, lut, kind).to(query.dtype)


def hf_attention(module, query, key, value, attention_mask, scaling=None, softcap=None, *, lut, kind="int8",
                 **kwargs):
    """Hugging Face ``AttentionInterface`` adapter (same contract as eager attention)."""
    groups = query.shape[1] // key.shape[1]
    key, value = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    out = approx_attention(query, key, value, lut, attention_mask, scaling, softcap, kind)
    return out.transpose(1, 2).contiguous(), None


def approximate_model(model, lut, layers=("*",), skip=("lm_head",), attention=True, kind="int8"):
    """Swap selected nn.Linear for LutLinear; route Hugging Face attention through the LUT.

    ``layers``/``skip`` are fnmatch patterns on module names. Updating ``lut``
    in place (``lut.copy_(...)``) switches every patched layer to a new multiplier.
    Returns the names of the replaced Linear layers.
    """
    names = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and any(fnmatch(name, p) for p in layers)
        and not any(fnmatch(name, p) for p in skip)
    ]
    for name in names:
        parent, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent), attribute, LutLinear(model.get_submodule(name), lut, kind))
    if attention:
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface, eager_mask

        # ponytail: process-global registration, one LUT per process; key by LUT if that changes.
        AttentionInterface.register("approx_lut", partial(hf_attention, lut=lut, kind=kind))
        # Without this, transformers passes no mask at all to unknown backends.
        AttentionMaskInterface.register("approx_lut", eager_mask)
        model.set_attn_implementation("approx_lut")
    return names
