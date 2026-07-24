"""Benchmark KV cache INT8 quantization vs FP16."""

import argparse
import json
import time
from pathlib import Path

import torch

from ..cuda import dequantize_kv_int8, quantize_kv_int8


def _time(fn, *args, repeats=50, warmup=10, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def bench_kv_quant():
    results = []
    for C, N in [(32, 4096), (64, 8192), (128, 16384)]:
        x = torch.randn(C, N, device="cuda", dtype=torch.float16)

        t_quant = _time(quantize_kv_int8, x)
        q, scale = quantize_kv_int8(x)
        t_dequant = _time(dequantize_kv_int8, q, scale)

        fp16_bytes = C * N * 2
        int8_bytes = C * N * 1 + C * 4
        results.append(
            {
                "C": C,
                "N": N,
                "quant_ms": t_quant * 1000,
                "dequant_ms": t_dequant * 1000,
                "fp16_mb": fp16_bytes / 1e6,
                "int8_mb": int8_bytes / 1e6,
                "compression": fp16_bytes / int8_bytes,
            }
        )
        print(
            f"kv_quant C={C:3d} N={N:6d} quant={t_quant*1000:6.2f}ms dequant={t_dequant*1000:6.2f}ms "
            f"fp16={fp16_bytes/1e6:.1f}MB int8={int8_bytes/1e6:.1f}MB compression={fp16_bytes/int8_bytes:.1f}x"
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/ops_kv_quant.json")
    args = parser.parse_args()

    results = bench_kv_quant()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
