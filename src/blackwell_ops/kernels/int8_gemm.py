"""INT8 weight-only GEMM Triton kernel for SM120.

Quantizes weights per output channel to int8, computes y = x @ W.T * scale.
This is a W8A16 kernel: activations fp16, weights int8, accumulation fp32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_gemm_kernel(
    X_ptr,
    W_ptr,
    S_ptr,
    Y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + (k + offs_k[None, :]) * stride_xk,
            mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K),
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            W_ptr + offs_n[None, :] * stride_wn + (k + offs_k[:, None]) * stride_wk,
            mask=(offs_n[None, :] < N) & ((k + offs_k[:, None]) < K),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(x, w)

    scale = tl.load(S_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc * scale[None, :]
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yk,
        acc.to(Y_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def quantize_weight_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric int8 quantization."""
    w_abs_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = w_abs_max / 127.0
    w_int8 = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return w_int8, scale.squeeze(-1)


def int8_gemm(
    x: torch.Tensor, w_int8: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """
    x: [M, K] fp16
    w_int8: [N, K] int8
    scale: [N] fp16/fp32
    returns: [M, N] fp16
    """
    M, K = x.shape
    N = w_int8.shape[0]
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)

    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _int8_gemm_kernel[grid](
        x,
        w_int8,
        scale,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_int8.stride(0),
        w_int8.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return y
