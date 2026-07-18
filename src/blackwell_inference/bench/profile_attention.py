"""torch.profiler wrapper for attention backends (ncu/nsys unavailable on WSL2+SM120)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from ..attention import forward as attention_forward


def profile_attention(backend: str, batch: int = 1, heads: int = 8, seq_len: int = 1024, head_dim: int | None = None) -> dict:
    if head_dim is None:
        head_dim = 128 if backend == "mini" else 64
    if backend in ("kda", "linear"):
        causal = True
    else:
        causal = False

    q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16) / math.sqrt(head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16) / math.sqrt(head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

    attention_forward(backend, q, k, v, causal=causal)  # warmup

    out_dir = Path(__file__).resolve().parent.parent.parent.parent / "benchmarks"
    out_dir.mkdir(exist_ok=True)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            attention_forward(backend, q, k, v, causal=causal)
            torch.cuda.synchronize()

    trace_path = out_dir / f"profile_{backend}_n{seq_len}.json"
    prof.export_chrome_trace(str(trace_path))

    rows = []
    for evt in prof.key_averages():
        if evt.self_device_time_total > 0:
            rows.append({
                "name": evt.key[:120],
                "calls": evt.count,
                "self_cuda_us": evt.self_device_time_total,
                "total_cuda_us": evt.device_time_total,
            })
    rows.sort(key=lambda r: -r["self_cuda_us"])
    summary = {
        "backend": backend,
        "batch": batch,
        "heads": heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "causal": causal,
        "total_kernel_ms": sum(r["self_cuda_us"] for r in rows) / 1000,
        "top_kernels": rows[:25],
    }
    summary_path = out_dir / f"profile_summary_{backend}_n{seq_len}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"trace": str(trace_path), "summary": str(summary_path), "data": summary}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=None)
    args = parser.parse_args()
    result = profile_attention(args.backend, args.batch, args.heads, args.seq_len, args.head_dim)
    print(result["trace"])
    print(result["summary"])
