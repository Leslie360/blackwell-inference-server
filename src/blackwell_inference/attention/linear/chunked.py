"""
Chunked parallel causal linear attention for SM120 (RTX 5070 Ti).

Two-pass chunked algorithm (plus a light scan in between), all in Triton:

  pass1 (_chunk_states_kernel):   per chunk c, S_c = sum_{j in c} k_j^T v_j
  scan  (_state_scan_kernel):     exclusive prefix  H_c = sum_{c' < c} S_{c'}
  pass2 (_chunk_fwd_o_kernel):    O_i = sum_{j<=i, j in c(i)} (q_i.k_j) v_j  +  q_i @ H_{c(i)}

Optional GLA-style scalar decay gate log_g (log-space, <= 0, per kv-head):
  within chunk: A_ij *= exp(lg_i - lg_j)        (lg = in-chunk inclusive cumsum)
  chunk state:  S_c  = sum_j exp(G_c - lg_j) k_j^T v_j   (G_c = sum of chunk)
  scan:         H_{c+1} = exp(G_c) * H_c + S_c
  inter-chunk:  O_i  += (q_i * exp(lg_i)) @ H_{c(i)}

Everything accumulates in fp32; states H/S are fp32. Dots that involve an
fp32-computed operand (A @ V, q @ H) run with input_precision="tf32" to keep
tensor-core speed without bf16 rounding of the attention matrix.

chunk_size is a wrapper-level parameter (64 or 128) shared by all three
kernels; autotune only covers num_warps/num_stages so the chunking can never
disagree between kernels.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# pass 1: per-chunk states S_c = sum_{j in c} decay(j->end) * k_j^T v_j
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"num_warps": 4, "num_stages": 2}),
        triton.Config({"num_warps": 4, "num_stages": 3}),
        triton.Config({"num_warps": 8, "num_stages": 2}),
        triton.Config({"num_warps": 8, "num_stages": 3}),
    ],
    key=["BLOCK_C", "D_K", "D_V"],
)
@triton.jit
def _chunk_states_kernel(
    K_ptr,
    V_ptr,
    S_ptr,
    LOGG_ptr,
    GSUM_ptr,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    H_KV,
    N,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_C: tl.constexpr,
    GATED: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_dk = tl.arange(0, D_K)
    offs_dv = tl.arange(0, D_V)
    mask_c = offs_c < N

    b = pid_bh // H_KV
    h = pid_bh % H_KV

    k_ptrs = (
        K_ptr
        + b * stride_kb
        + h * stride_kh
        + offs_c[:, None] * stride_kn
        + offs_dk[None, :] * stride_kd
    )
    v_ptrs = (
        V_ptr
        + b * stride_vb
        + h * stride_vh
        + offs_c[:, None] * stride_vn
        + offs_dv[None, :] * stride_vd
    )
    k = tl.load(k_ptrs, mask=mask_c[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_c[:, None], other=0.0)

    if GATED:
        # decay from position j to chunk end: exp(G_c - lg_j)
        log_g = tl.load(LOGG_ptr + pid_bh * N + offs_c, mask=mask_c, other=0.0)
        lg = tl.cumsum(log_g, 0)
        g_sum = tl.sum(log_g, 0)
        tl.store(GSUM_ptr + pid_bh * tl.num_programs(1) + pid_c, g_sum)
        k = (k.to(tl.float32) * tl.exp(g_sum - lg)[:, None]).to(k.dtype)

    # S_c [D_K, D_V] fp32
    s = tl.dot(tl.trans(k), v)
    s_ptrs = (
        S_ptr
        + pid_bh * (tl.num_programs(1) * D_K * D_V)
        + pid_c * (D_K * D_V)
        + offs_dk[:, None] * D_V
        + offs_dv[None, :]
    )
    tl.store(s_ptrs, s)


# ---------------------------------------------------------------------------
# scan: exclusive prefix over chunks, H_c = sum_{c'<c} decayed S_{c'}
# grid (B*H_KV, (D_K/BS)*(D_V/BS)); sequential over NC but tiny steps
# ---------------------------------------------------------------------------
@triton.jit
def _state_scan_kernel(
    S_ptr,
    H_ptr,
    GSUM_ptr,
    NC,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_S: tl.constexpr,
    GATED: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)
    num_v_tiles = D_V // BLOCK_S
    tile_k = pid_t // num_v_tiles
    tile_v = pid_t % num_v_tiles

    offs_k = tile_k * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_v = tile_v * BLOCK_S + tl.arange(0, BLOCK_S)

    base = pid_bh * (NC * D_K * D_V)
    run = tl.zeros((BLOCK_S, BLOCK_S), dtype=tl.float32)
    for c in range(NC):
        offs = base + c * (D_K * D_V) + offs_k[:, None] * D_V + offs_v[None, :]
        tl.store(H_ptr + offs, run)  # exclusive prefix: state BEFORE chunk c
        s = tl.load(S_ptr + offs)
        if GATED:
            g = tl.load(GSUM_ptr + pid_bh * NC + c)
            run = run * tl.exp(g) + s
        else:
            run += s


# ---------------------------------------------------------------------------
# pass 2: outputs. O = tril(decay * QK^T) @ V  +  (decay * Q) @ H_c
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"num_warps": 4, "num_stages": 2}),
        triton.Config({"num_warps": 4, "num_stages": 3}),
        triton.Config({"num_warps": 8, "num_stages": 2}),
        triton.Config({"num_warps": 8, "num_stages": 3}),
    ],
    key=["BLOCK_C", "D_K", "D_V"],
)
@triton.jit
def _chunk_fwd_o_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    H_ptr,
    O_ptr,
    LOGG_ptr,
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
    H_Q,
    GROUP,
    N,
    NC,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_C: tl.constexpr,
    GATED: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_c = tl.program_id(1)

    b = pid_bh // H_Q
    h_q = pid_bh % H_Q
    h_kv = h_q // GROUP

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_dk = tl.arange(0, D_K)
    offs_dv = tl.arange(0, D_V)
    mask_c = offs_c < N

    q_ptrs = (
        Q_ptr
        + b * stride_qb
        + h_q * stride_qh
        + offs_c[:, None] * stride_qn
        + offs_dk[None, :] * stride_qd
    )
    k_ptrs = (
        K_ptr
        + b * stride_kb
        + h_kv * stride_kh
        + offs_c[:, None] * stride_kn
        + offs_dk[None, :] * stride_kd
    )
    v_ptrs = (
        V_ptr
        + b * stride_vb
        + h_kv * stride_vh
        + offs_c[:, None] * stride_vn
        + offs_dv[None, :] * stride_vd
    )
    q = tl.load(q_ptrs, mask=mask_c[:, None], other=0.0)
    k = tl.load(k_ptrs, mask=mask_c[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_c[:, None], other=0.0)

    # intra-chunk: A_ij = (q_i . k_j) * decay(i<-j), j <= i
    a = tl.dot(q, tl.trans(k))
    causal = tl.arange(0, BLOCK_C)[:, None] >= tl.arange(0, BLOCK_C)[None, :]
    if GATED:
        log_g = tl.load(
            LOGG_ptr + (b * (H_Q // GROUP) + h_kv) * N + offs_c, mask=mask_c, other=0.0
        )
        lg = tl.cumsum(log_g, 0)
        a = a * tl.exp(lg[:, None] - lg[None, :])
    a = tl.where(causal, a, 0.0)
    o = tl.dot(a, v.to(tl.float32), input_precision="tf32")

    # inter-chunk: O_i += (q_i * decay(chunk_start->i)) @ H_c
    h_ptrs = (
        H_ptr
        + (b * (H_Q // GROUP) + h_kv) * (NC * D_K * D_V)
        + pid_c * (D_K * D_V)
        + offs_dk[:, None] * D_V
        + offs_dv[None, :]
    )
    h = tl.load(h_ptrs)
    if GATED:
        q2 = q.to(tl.float32) * tl.exp(lg)[:, None]
    else:
        q2 = q.to(tl.float32)
    o += tl.dot(q2, h, input_precision="tf32")

    o_ptrs = (
        O_ptr
        + b * stride_ob
        + h_q * stride_oh
        + offs_c[:, None] * stride_on
        + offs_dv[None, :] * stride_od
    )
    tl.store(o_ptrs, o.to(O_ptr.dtype.element_ty), mask=mask_c[:, None])


def linear_attention_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_g: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    q:     [B, H_Q, N, D_K]  bf16/fp16
    k, v:  [B, H_KV, N, D]   (H_Q % H_KV == 0, GQA via head grouping)
    log_g: optional [B, H_KV, N] fp32 log-decay (<= 0), GLA-style scalar gate
    returns O: [B, H_Q, N, D_V] (same dtype as q)
    """
    B, H_Q, N, D_K = q.shape
    H_KV = k.shape[1]
    D_V = v.shape[-1]
    assert H_Q % H_KV == 0
    assert D_K in (64, 128) and D_V in (64, 128)
    group = H_Q // H_KV
    gated = log_g is not None

    o = torch.empty(B, H_Q, N, D_V, device=q.device, dtype=q.dtype)
    NC = triton.cdiv(N, chunk_size)

    # chunk states + exclusive prefix states, fp32, contiguous
    states = torch.empty(B * H_KV, NC, D_K, D_V, device=q.device, dtype=torch.float32)
    hstates = torch.empty_like(states)
    if gated:
        log_g = log_g.float().contiguous()
        gsums = torch.empty(B * H_KV, NC, device=q.device, dtype=torch.float32)
    else:
        gsums = states  # unused dummy

    _chunk_states_kernel[(B * H_KV, NC)](
        k,
        v,
        states,
        log_g if gated else k,
        gsums,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        H_KV,
        N,
        D_K=D_K,
        D_V=D_V,
        BLOCK_C=chunk_size,
        GATED=gated,
    )
    BLOCK_S = 16
    scan_grid = (B * H_KV, (D_K // BLOCK_S) * (D_V // BLOCK_S))
    _state_scan_kernel[scan_grid](
        states,
        hstates,
        gsums,
        NC,
        D_K=D_K,
        D_V=D_V,
        BLOCK_S=BLOCK_S,
        GATED=gated,
    )
    _chunk_fwd_o_kernel[(B * H_Q, NC)](
        q,
        k,
        v,
        hstates,
        o,
        log_g if gated else q,
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
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        H_Q,
        group,
        N,
        NC,
        D_K=D_K,
        D_V=D_V,
        BLOCK_C=chunk_size,
        GATED=gated,
    )
    return o
