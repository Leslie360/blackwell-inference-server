"""Benchmark INT8 weight-only GEMM vs FP16 matmul."""

import argparse
import json
import time
from pathlib import Path

import torch

from ..kernels import int8_gemm, quantize_weight_int8


def _time(fn, *args, repeats=50, warmup=10, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def bench_int8_gemm():
    results = []
    for M, K, N in [(1024, 1024, 1024), (4096, 4096, 4096), (8192, 4096, 14336)]:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        w = torch.randn(N, K, device="cuda", dtype=torch.float16)
        w_int8, scale = quantize_weight_int8(w)

        t_int8 = _time(int8_gemm, x, w_int8, scale)
        t_fp16 = _time(lambda: x @ w.t())
        flops = 2.0 * M * N * K * 1e-12
        results.append(
            {
                "M": M,
                "K": K,
                "N": N,
                "int8_ms": t_int8 * 1000,
                "fp16_ms": t_fp16 * 1000,
                "speedup": t_fp16 / t_int8,
                "int8_tflops": flops / t_int8,
                "fp16_tflops": flops / t_fp16,
            }
        )
        print(
            f"int8_gemm M={M:5d} K={K:5d} N={N:5d} int8={t_int8*1000:6.2f}ms fp16={t_fp16*1000:6.2f}ms speedup={t_fp16/t_int8:.2f}x"
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/ops_int8_gemm.json")
    args = parser.parse_args()

    results = bench_int8_gemm()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
