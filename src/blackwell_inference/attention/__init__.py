"""Attention backends for consumer Blackwell GPUs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def forward(backend: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """
    Unified attention dispatch.

    backend: one of "sdpa", "linear", "kda", "mini"
    q, k, v: [B, H, N, D]
    """
    if backend == "sdpa":
        return sdpa(q, k, v, causal=causal)
    if backend == "linear":
        from .linear import linear_attention_chunked
        return linear_attention_chunked(q, k, v, chunk_size=128)
    if backend == "kda":
        from .kda import forward as kda_forward
        return kda_forward(q, k, v)
    if backend == "mini":
        from .mini import forward as mini_forward
        return mini_forward(q, k, v)
    raise ValueError(f"unknown backend: {backend}")


__all__ = ["forward", "sdpa"]
