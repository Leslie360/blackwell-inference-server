"""Benchmark blackwell-ops kernels vs PyTorch reference."""

import argparse
import json
import time
from pathlib import Path

import torch

from ..kernels import precompute_cos_sin, rmsnorm, rope


def _time(fn, *args, repeats=50, warmup=10, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def bench_rmsnorm():
    results = []
    for M, N in [(1024, 512), (4096, 1024), (8192, 4096)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        w = torch.randn(N, device="cuda", dtype=torch.float16)

        def ref():
            x_f = x.float()
            return (
                x_f
                / torch.sqrt((x_f * x_f).mean(dim=-1, keepdim=True) + 1e-6)
                * w.float()
            )

        t_ours = _time(rmsnorm, x, w)
        t_ref = _time(ref)
        bytes_moved = 2 * M * N * 2 * 2  # read x, write y, fp16
        results.append(
            {
                "M": M,
                "N": N,
                "ours_ms": t_ours * 1000,
                "ref_ms": t_ref * 1000,
                "speedup": t_ref / t_ours,
                "ours_gbs": bytes_moved / t_ours / 1e9,
            }
        )
        print(
            f"rmsnorm M={M:5d} N={N:5d} ours={t_ours*1000:6.2f}ms ref={t_ref*1000:6.2f}ms speedup={t_ref/t_ours:.2f}x"
        )
    return results


def bench_rope():
    results = []
    for B, H, N, D in [(8, 16, 512, 64), (4, 32, 1024, 128), (1, 32, 4096, 128)]:
        x = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        cos, sin = precompute_cos_sin(N, D, device="cuda")

        def ref():
            x1 = x[..., : D // 2].float()
            x2 = x[..., D // 2 :].float()
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        t_ours = _time(rope, x, cos, sin)
        t_ref = _time(ref)
        bytes_moved = 2 * B * H * N * D * 2 * 2
        results.append(
            {
                "B": B,
                "H": H,
                "N": N,
                "D": D,
                "ours_ms": t_ours * 1000,
                "ref_ms": t_ref * 1000,
                "speedup": t_ref / t_ours,
                "ours_gbs": bytes_moved / t_ours / 1e9,
            }
        )
        print(
            f"rope B={B} H={H:2d} N={N:5d} D={D:3d} ours={t_ours*1000:6.2f}ms ref={t_ref*1000:6.2f}ms speedup={t_ref/t_ours:.2f}x"
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/ops_basic.json")
    args = parser.parse_args()

    print("=== RMSNorm ===")
    rms = bench_rmsnorm()
    print("\n=== RoPE ===")
    rp = bench_rope()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"rmsnorm": rms, "rope": rp}, f, indent=2)
    print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
