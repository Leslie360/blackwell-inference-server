"""Rotary Position Embedding (RoPE) Triton kernel for SM120.

Applies: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos) for each position.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fwd_kernel(
    X_ptr,
    COS_ptr,
    SIN_ptr,
    Y_ptr,
    stride_xb,
    stride_xh,
    stride_xn,
    stride_xd,
    stride_yb,
    stride_yh,
    stride_yn,
    stride_yd,
    H,
    N,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_d = tl.arange(0, BLOCK_D)
    half = D // 2
    mask = offs_d < half

    x_ptr = X_ptr + pid_b * stride_xb + pid_h * stride_xh + pid_n * stride_xn
    y_ptr = Y_ptr + pid_b * stride_yb + pid_h * stride_yh + pid_n * stride_yn

    x1 = tl.load(x_ptr + offs_d * stride_xd, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + (offs_d + half) * stride_xd, mask=mask, other=0.0).to(
        tl.float32
    )
    cos = tl.load(COS_ptr + pid_n * half + offs_d, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(SIN_ptr + pid_n * half + offs_d, mask=mask, other=0.0).to(tl.float32)

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    tl.store(y_ptr + offs_d * stride_yd, y1.to(Y_ptr.dtype.element_ty), mask=mask)
    tl.store(
        y_ptr + (offs_d + half) * stride_yd, y2.to(Y_ptr.dtype.element_ty), mask=mask
    )


def precompute_cos_sin(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin for RoPE."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)
    )
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)  # [seq_len, dim/2]
    return freqs.cos(), freqs.sin()


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: [B, H, N, D] fp16/bf16/fp32, D must be even
    cos/sin: [N, D/2] fp32
    returns: [B, H, N, D]
    """
    B, H, N, D = x.shape
    assert D % 2 == 0
    y = torch.empty_like(x)
    BLOCK_D = triton.next_power_of_2(D // 2)
    grid = (B, H, N)
    _rope_fwd_kernel[grid](
        x,
        cos,
        sin,
        y,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        H,
        N,
        D,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return y
