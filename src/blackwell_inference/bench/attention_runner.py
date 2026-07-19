"""Benchmark harness for attention backends."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..attention import forward as attention_forward


@dataclass
class AttentionResult:
    backend: str
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    causal: bool
    latency_ms: float
    tflops: float
    max_err: float | None


def _create_tensors(
    B: int, H: int, N: int, D: int, dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype) / math.sqrt(D)
    k = torch.randn(B, H, N, D, device="cuda", dtype=dtype) / math.sqrt(D)
    v = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    return q, k, v


def _time(fn, *args, repeats: int = 20, warmup: int = 5, **kwargs) -> float:
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def benchmark_attention(
    backend: str,
    batch: int = 1,
    heads: int = 8,
    seq_len: int = 1024,
    head_dim: int | None = None,
    dtype: torch.dtype = torch.float16,
    causal: bool = False,
    repeats: int = 20,
) -> AttentionResult:
    if head_dim is None:
        head_dim = 128 if backend in ("mini",) else 64
    # KDA and linear attention are causal-only kernels
    if backend in ("kda", "linear"):
        causal = True
    q, k, v = _create_tensors(batch, heads, seq_len, head_dim, dtype)

    out = attention_forward(backend, q, k, v, causal=causal)
    if backend == "linear":
        # linear attention = unnormalized causal sum, not softmax
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~mask, 0.0)
        ref = torch.matmul(scores, v.float()).to(q.dtype)
    else:
        ref = attention_forward("sdpa", q, k, v, causal=causal)
    atol = 0.05 * ref.abs().max().item()
    err = (out.float() - ref.float()).abs().max().item()

    # causal attention FLOPs ~2*B*H*N^2*D; non-causal ~4*B*H*N^2*D
    flops = (
        (2.0 if causal else 4.0) * batch * heads * seq_len * seq_len * head_dim * 1e-12
    )
    t = _time(
        lambda: attention_forward(backend, q, k, v, causal=causal), repeats=repeats
    )
    return AttentionResult(
        backend=backend,
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        causal=causal,
        latency_ms=t * 1000,
        tflops=flops / t,
        max_err=err if err < atol else err,
    )


def run_grid(
    backends: list[str],
    seq_lens: list[int],
    batch: int = 1,
    heads: int = 8,
    causal: bool = False,
    repeats: int = 20,
) -> list[AttentionResult]:
    results = []
    for backend in backends:
        for N in seq_lens:
            try:
                res = benchmark_attention(
                    backend, batch, heads, N, causal=causal, repeats=repeats
                )
            except Exception as e:
                print(f"[warn] {backend} N={N}: {e}")
                continue
            results.append(res)
            print(
                f"{backend:8s} N={N:5d} {res.latency_ms:8.3f} ms  {res.tflops:6.1f} TFLOPS  err={res.max_err:.2e}"
            )
    return results


def save_results(results: list[AttentionResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
