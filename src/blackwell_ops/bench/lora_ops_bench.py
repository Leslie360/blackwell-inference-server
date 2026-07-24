"""Benchmark LoRA CUDA kernels vs PyTorch reference."""

import argparse
import json
import time
from pathlib import Path

import torch

from ..cuda import lora_delta


def _time(fn, *args, repeats=50, warmup=10, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def bench_lora_ops():
    results = []
    for r, K, N in [(16, 1024, 2048), (128, 4096, 4096), (128, 4096, 14336)]:
        A = torch.randn(r, K, device="cuda", dtype=torch.float16)
        B = torch.randn(N, r, device="cuda", dtype=torch.float16)
        scaling = 2.0

        t_ours = _time(lora_delta, A, B, scaling)
        t_ref = _time(lambda: (B.float() @ A.float()) * scaling)
        flops = 2.0 * N * K * r * 1e-12
        results.append(
            {
                "r": r,
                "K": K,
                "N": N,
                "ours_ms": t_ours * 1000,
                "ref_ms": t_ref * 1000,
                "speedup": t_ref / t_ours,
                "ours_tflops": flops / t_ours,
                "ref_tflops": flops / t_ref,
            }
        )
        print(
            f"lora_delta r={r:3d} K={K:5d} N={N:5d} ours={t_ours*1000:6.2f}ms ref={t_ref*1000:6.2f}ms speedup={t_ref/t_ours:.2f}x"
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/ops_lora.json")
    args = parser.parse_args()

    results = bench_lora_ops()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
