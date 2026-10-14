"""LUT-approximate int8 matmuls for transformers: Linear layers and attention.

Every multiply in a GEMM goes through the multiplier LUT: the flat 65536-entry
int32 tensor from ``load_lut(path, 'int8')``, indexed
``(a + 128) * 256 + (b + 128)`` with ``a`` the activation (row) operand.
Pretrained models have no QAT checkpoint, so quantization is symmetric int8:
weights per output channel (once), activations per token (every call), and
both attention operands per row/column. Elementwise products (RoPE, norms,
GLU gates) stay exact.

On CUDA with a sign-symmetric LUT, matmuls run as one batched LUT-GEMM launch
(``backend.lut_kernel``); otherwise they loop over ``ops.gemm_int8``.
"""

from __future__ import annotations

from fnmatch import fnmatch
from functools import partial

import torch
from torch import nn

from .backend import lut_kernel, ops


def quantize(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.abs().amax(dim, keepdim=True).clamp_min(1e-12) / 127
    return torch.round(x / scale).clamp(-127, 127).to(torch.int8), scale


def lut_matmul(a, b, lut, symmetric=False) -> torch.Tensor:
    """int8 ``a @ b`` through the LUT, batched over broadcast leading dims.

    ``symmetric``: both operands are in [-127, 127] (``quantize`` output), so a
    sign-symmetric CUDA LUT can run as one batched lut_gemm launch (bit-identical).
    """
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = a.expand(*batch, *a.shape[-2:]).reshape(-1, *a.shape[-2:])
    b = b.expand(*batch, *b.shape[-2:]).reshape(-1, *b.shape[-2:])
    t = lut_kernel.table(lut, lut_kernel.SIGNMAG) if symmetric else None
    if t is not None:
        out = lut_kernel.batched(a, b, t).float()
    else:
        out = torch.stack([ops.gemm_int8(x.contiguous(), y.contiguous(), lut).float() for x, y in zip(a, b)])
    return out.reshape(*batch, *out.shape[-2:])


def approx_matmul(a, b, lut) -> torch.Tensor:
    """Float ``a @ b`` with a quantized per row and b per column."""
    a_q, a_scale = quantize(a.float(), -1)
    b_q, b_scale = quantize(b.float(), -2)
    return lut_matmul(a_q, b_q, lut, symmetric=True) * a_scale * b_scale


class LutLinear(nn.Module):
    def __init__(self, linear: nn.Linear, lut: torch.Tensor) -> None:
        super().__init__()
        weight_q, weight_scale = quantize(linear.weight.detach().float().t(), 0)
        self.register_buffer("weight_q", weight_q.contiguous())
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer(
            "bias", None if linear.bias is None else linear.bias.detach().float()
        )
        self.lut = lut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q, x_scale = quantize(x.float().reshape(-1, x.shape[-1]), -1)
        y = lut_matmul(x_q, self.weight_q, self.lut, True) * x_scale * self.weight_scale
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], -1).to(x.dtype)


def approx_attention(query, key, value, lut, attn_mask=None, scale=None, softcap=None):
    """softmax(Q K^T * scale [+ additive mask]) V with both matmuls through the LUT."""
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    scores = approx_matmul(query, key.transpose(-2, -1), lut) * scale
    if softcap is not None:
        scores = torch.tanh(scores / softcap) * softcap
    if attn_mask is not None:
        scores = scores + attn_mask
    return approx_matmul(scores.softmax(-1), value, lut).to(query.dtype)


def hf_attention(module, query, key, value, attention_mask, scaling=None, softcap=None, *, lut, **kwargs):
    """Hugging Face ``AttentionInterface`` adapter (same contract as eager attention)."""
    groups = query.shape[1] // key.shape[1]
    key, value = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    out = approx_attention(query, key, value, lut, attention_mask, scaling, softcap)
    return out.transpose(1, 2).contiguous(), None


def approximate_model(model, lut, layers=("*",), skip=("lm_head",), attention=True):
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
        setattr(model.get_submodule(parent), attribute, LutLinear(model.get_submodule(name), lut))
    if attention:
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface, eager_mask

        # ponytail: process-global registration, one LUT per process; key by LUT if that changes.
        AttentionInterface.register("approx_lut", partial(hf_attention, lut=lut))
        # Without this, transformers passes no mask at all to unknown backends.
        AttentionMaskInterface.register("approx_lut", eager_mask)
        model.set_attn_implementation("approx_lut")
    return names
