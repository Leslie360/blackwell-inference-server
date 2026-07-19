"""RMSNorm Triton kernel for SM120.

Computes: y = x / sqrt(mean(x^2) + eps) * weight
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    stride_x_row,
    stride_y_row,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Y_ptr + row * stride_y_row + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: [M, N] fp16/bf16/fp32
    weight: [N]
    returns: [M, N]
    """
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    _rmsnorm_fwd_kernel[(M,)](
        x,
        weight,
        y,
        x.stride(0),
        y.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return y
