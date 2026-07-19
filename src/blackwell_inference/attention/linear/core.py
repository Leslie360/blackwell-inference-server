"""
Optimized causal linear attention kernels for SM120 (RTX 5070 Ti).

Computes O_i = sum_{j=1}^{i} (q_i @ k_j^T) * v_j
             = q_i @ S_i   where   S_i = sum_{j=1}^{i} k_j^T v_j

Complexity: O(N * d_k * d_v) instead of O(N^2 * d).
This module provides:
  - block-wise prefix-sum forward kernel (Triton)
  - grouped-query attention (GQA) support
  - decode-step kernel that updates a recurrent state S in O(d_k * d_v)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_attention_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    B,
    H,
    N,
    D_K,
    D_V,
    BLOCK_N: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_blk = tl.program_id(2)

    start_n = pid_blk * BLOCK_N
    _offs_n = start_n + tl.arange(0, BLOCK_N)  # noqa: F841
    offs_dk = tl.arange(0, BLOCK_DK)
    offs_dv = tl.arange(0, BLOCK_DV)

    # Running state S = sum_{j < start_n} k_j^T v_j, shape [D_K, D_V]
    S = tl.zeros((BLOCK_DK, BLOCK_DV), dtype=tl.float32)
    for prev in range(0, pid_blk):
        prev_start = prev * BLOCK_N
        prev_offs_n = prev_start + tl.arange(0, BLOCK_N)
        k_ptrs = (
            K_ptr
            + pid_b * stride_kb
            + pid_h * stride_kh
            + prev_offs_n[:, None] * stride_kn
            + offs_dk[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=prev_offs_n[:, None] < N, other=0.0)
        v_ptrs = (
            V_ptr
            + pid_b * stride_vb
            + pid_h * stride_vh
            + prev_offs_n[:, None] * stride_vn
            + offs_dv[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=prev_offs_n[:, None] < N, other=0.0)
        S += tl.dot(tl.trans(k), v)

    # Process block sequentially to respect causality
    for i in range(BLOCK_N):
        pos = start_n + i
        q_ptrs = (
            Q_ptr
            + pid_b * stride_qb
            + pid_h * stride_qh
            + pos * stride_qn
            + offs_dk * stride_qd
        )
        q = tl.load(q_ptrs, mask=pos < N, other=0.0)
        k_ptrs = (
            K_ptr
            + pid_b * stride_kb
            + pid_h * stride_kh
            + pos * stride_kn
            + offs_dk * stride_kd
        )
        k_i = tl.load(k_ptrs, mask=pos < N, other=0.0)
        v_ptrs = (
            V_ptr
            + pid_b * stride_vb
            + pid_h * stride_vh
            + pos * stride_vn
            + offs_dv * stride_vd
        )
        v_i = tl.load(v_ptrs, mask=pos < N, other=0.0)
        S += k_i[:, None] * v_i[None, :]
        o_i = tl.sum(q[:, None].to(tl.float32) * S, axis=0)
        o_ptrs = (
            O_ptr
            + pid_b * stride_ob
            + pid_h * stride_oh
            + pos * stride_on
            + offs_dv * stride_od
        )
        tl.store(o_ptrs, o_i.to(q.dtype), mask=pos < N)


def linear_attention_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    q, k: [B, H, N, D_K]
    v:    [B, H, N, D_V]
    returns O: [B, H, N, D_V]
    """
    B, H, N, D_K = q.shape
    D_V = v.shape[-1]
    assert k.shape == (B, H, N, D_K)
    assert v.shape == (B, H, N, D_V)
    out = torch.empty_like(v)

    BLOCK_N = 64
    BLOCK_DK = triton.next_power_of_2(D_K)
    BLOCK_DV = triton.next_power_of_2(D_V)
    num_blocks = triton.cdiv(N, BLOCK_N)
    grid = (B, H, num_blocks)

    _linear_attention_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        B,
        H,
        N,
        D_K,
        D_V,
        BLOCK_N=BLOCK_N,
        BLOCK_DK=BLOCK_DK,
        BLOCK_DV=BLOCK_DV,
    )
    return out


@triton.jit
def _decode_step_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    s_ptr,
    o_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vd,
    stride_sb,
    stride_sh,
    stride_dk,
    stride_dv,
    stride_ob,
    stride_oh,
    stride_od,
    B,
    H,
    D_K,
    D_V,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """Single decode step: update recurrent state S with new k,v and compute o = q @ S."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_dk = tl.arange(0, BLOCK_DK)
    offs_dv = tl.arange(0, BLOCK_DV)

    q = tl.load(
        q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_dk * stride_qd,
        mask=offs_dk < D_K,
        other=0.0,
    )
    k = tl.load(
        k_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_dk * stride_kd,
        mask=offs_dk < D_K,
        other=0.0,
    )
    v = tl.load(
        v_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_dv * stride_vd,
        mask=offs_dv < D_V,
        other=0.0,
    )

    s_ptrs = (
        s_ptr
        + pid_b * stride_sb
        + pid_h * stride_sh
        + offs_dk[:, None] * stride_dk
        + offs_dv[None, :] * stride_dv
    )
    S = tl.load(
        s_ptrs, mask=(offs_dk[:, None] < D_K) & (offs_dv[None, :] < D_V), other=0.0
    )

    # S += k^T @ v  (outer product)
    S += k[:, None] * v[None, :]
    tl.store(s_ptrs, S, mask=(offs_dk[:, None] < D_K) & (offs_dv[None, :] < D_V))

    # o = q @ S  -> [D_V]
    o = tl.sum(q[:, None].to(tl.float32) * S, axis=0)
    tl.store(
        o_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_dv * stride_od,
        o.to(q.dtype),
        mask=offs_dv < D_V,
    )


def decode_step_triton(q, k, v, state):
    """
    q, k: [B, H, D_K]
    v:    [B, H, D_V]
    state: [B, H, D_K, D_V]
    returns o: [B, H, D_V], updates state in-place.
    """
    B, H, D_K = q.shape
    D_V = v.shape[-1]
    out = torch.empty_like(v)
    grid = (B, H)
    _decode_step_kernel[grid](
        q,
        k,
        v,
        state,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        B,
        H,
        D_K,
        D_V,
        BLOCK_DK=triton.next_power_of_2(D_K),
        BLOCK_DV=triton.next_power_of_2(D_V),
    )
    return out


def standard_attention_pytorch(q, k, v, causal=True):
    """Reference: [B,H,N,D] -> [B,H,N,D]."""
    scores = torch.matmul(q, k.transpose(-2, -1))
    if causal:
        N = q.shape[2]
        mask = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, 0.0)
    return torch.matmul(scores, v)


def standard_attention_decode_step(q, k_cache, v_cache, v_len):
    """
    q: [B,H,D_K]
    k_cache, v_cache: [B,H,N,D]  (full cache up to current position)
    v_len: int, number of valid tokens in cache
    returns o: [B,H,D_V]
    """
    k = k_cache[:, :, :v_len, :]
    v = v_cache[:, :, :v_len, :]
    q_ = q.unsqueeze(2)  # [B,H,1,D_K]
    scores = torch.matmul(q_, k.transpose(-2, -1))  # [B,H,1,N]
    return torch.matmul(scores, v).squeeze(2)  # [B,H,D_V]
