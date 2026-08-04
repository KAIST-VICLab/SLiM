# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import nn, Tensor


logger = logging.getLogger("slim")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float = None, init_proj_std: float = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


''' For skeleton-based self-supervised learning '''
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        # Precompute inverse frequencies (cached buffer)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_cos_sin(self, t_indices: torch.Tensor, device, dtype):
        freqs = torch.einsum("i,j->ij", t_indices.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, None, :].to(dtype=dtype)
        sin = emb.sin()[None, :, None, :].to(dtype=dtype)
        return cos, sin

    def forward(self, num_t: int, num_v: int, device, dtype):
        t = torch.arange(num_t, device=device, dtype=self.inv_freq.dtype)
        cos, sin = self._compute_cos_sin(t, device, dtype)  # shape: (1, T, 1, D)
        cos = cos.repeat_interleave(num_v, dim=1)
        sin = sin.repeat_interleave(num_v, dim=1)
        return cos, sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_fp32 = q.float()
    k_fp32 = k.float()
    cos = cos.float()
    sin = sin.float()

    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)

    return q_embed.type_as(q), k_embed.type_as(k)


class MemEffAttention_RoPE(Attention):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            num_registers: int = 4,  # Additional arg
    ) -> None:
        # Base init
        super().__init__(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

        # Additional setup
        self.num_registers = num_registers
        head_dim = dim // num_heads
        self.rope = RotaryEmbedding(head_dim)

    def forward(self, x: Tensor, num_t: int, attn_bias=None) -> Tensor:
        """
        x: (B, N, C) where N = 1(CLS) + num_registers + (num_t * num_v)
        num_t: Temporal frames input
        """
        if not XFORMERS_AVAILABLE:
            raise AssertionError("xFormers is required for MemEffAttention_RoPE implementation.")

        B, N, C = x.shape

        # 1. QKV Calculation (using base class layer)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)  # (B, N, H, D)

        # ---------------------------------------------------------
        # RoPE Logic: Global(Center) vs Data(Temporal)
        # ---------------------------------------------------------

        # A. Grouping
        num_global = 1 + self.num_registers
        num_data = N - num_global

        if num_data % num_t != 0:
            raise ValueError(f"Data tokens ({num_data}) must be divisible by num_t ({num_t})")
        num_v = num_data // num_t

        # B. Generate Embeddings
        cos_data, sin_data = self.rope(num_t, num_v, x.device, x.dtype)

        # C. Split -> Apply -> Concat
        q_global, q_data = q[:, :num_global], q[:, num_global:]
        k_global, k_data = k[:, :num_global], k[:, num_global:]

        q_data, k_data = apply_rotary_pos_emb(q_data, k_data, cos_data, sin_data)

        q = torch.cat([q_global, q_data], dim=1)
        k = torch.cat([k_global, k_data], dim=1)

        # ---------------------------------------------------------

        # 2. xFormers Attention
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)

        # 3. Output Projection
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x