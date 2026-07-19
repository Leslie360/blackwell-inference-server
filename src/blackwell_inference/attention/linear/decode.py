"""
Single-step decode kernel for causal linear attention on SM120.

Maintains the recurrent state S [B, H_KV, D_K, D_V] (fp32, one state per
KV head, shared by the whole GQA group):

    S <- decay * S + k^T v          (O(D_K * D_V) per head)
    o_h = q_h @ S                   for each q-head h in the group

The update is strictly bandwidth-bound: one read + one write of S per step.
Everything is in-place on caller-provided buffers (state, out) with no
allocation inside the launch path, so it is CUDA-graph capturable.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_step_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    S_ptr,
    O_ptr,
    LOGG_ptr,
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
    stride_sdk,
    stride_sdv,
    stride_ob,
    stride_oh,
    stride_od,
    H_KV,
    GROUP,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    GATED: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # over B * H_KV
    pid_v = tl.program_id(1)  # over D_V / BLOCK_DV

    b = pid_bh // H_KV
    h_kv = pid_bh % H_KV

    offs_dk = tl.arange(0, D_K)
    offs_dv = pid_v * BLOCK_DV + tl.arange(0, BLOCK_DV)

    k = tl.load(K_ptr + b * stride_kb + h_kv * stride_kh + offs_dk * stride_kd)
    v = tl.load(V_ptr + b * stride_vb + h_kv * stride_vh + offs_dv * stride_vd)

    s_ptrs = (
        S_ptr
        + b * stride_sb
        + h_kv * stride_sh
        + offs_dk[:, None] * stride_sdk
        + offs_dv[None, :] * stride_sdv
    )
    s = tl.load(s_ptrs)

    if GATED:
        g = tl.load(LOGG_ptr + pid_bh)  # log_g [B, H_KV] contiguous
        s = s * tl.exp(g)
    s += k.to(tl.float32)[:, None] * v.to(tl.float32)[None, :]
    tl.store(s_ptrs, s)

    # one state per kv-head; loop over the q-heads sharing it
    for gi in range(GROUP):
        h_q = h_kv * GROUP + gi
        q = tl.load(Q_ptr + b * stride_qb + h_q * stride_qh + offs_dk * stride_qd)
        o = tl.sum(q.to(tl.float32)[:, None] * s, axis=0)
        tl.store(
            O_ptr + b * stride_ob + h_q * stride_oh + offs_dv * stride_od,
            o.to(O_ptr.dtype.element_ty),
        )


def decode_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor | None = None,
    log_g: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    q:     [B, H_Q, D_K]   bf16/fp16
    k:     [B, H_KV, D_K]
    v:     [B, H_KV, D_V]
    state: [B, H_KV, D_K, D_V] fp32, updated in place
    out:   optional preallocated [B, H_Q, D_V] (pass for CUDA graph capture)
    log_g: optional [B, H_KV] fp32 log-decay for this step
    returns out: [B, H_Q, D_V]
    """
    B, H_Q, D_K = q.shape
    H_KV = k.shape[1]
    D_V = v.shape[-1]
    assert H_Q % H_KV == 0
    assert state.shape == (B, H_KV, D_K, D_V) and state.dtype == torch.float32
    if out is None:
        out = torch.empty(B, H_Q, D_V, device=q.device, dtype=q.dtype)
    gated = log_g is not None
    if gated:
        log_g = log_g.float().contiguous()  # no-op when already fp32 contiguous

    BLOCK_DV = min(D_V, 64)
    grid = (B * H_KV, D_V // BLOCK_DV)
    _decode_step_kernel[grid](
        q,
        k,
        v,
        state,
        out,
        log_g if gated else q,
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
        H_KV,
        H_Q // H_KV,
        D_K=D_K,
        D_V=D_V,
        BLOCK_DV=BLOCK_DV,
        GATED=gated,
        num_warps=4,
    )
    return out
