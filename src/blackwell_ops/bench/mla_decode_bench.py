"""Benchmark MLA decode kernel vs PyTorch reference."""

import argparse
import json
import time
from pathlib import Path

import torch

from ..cuda import mla_decode


def _time(fn, *args, repeats=50, warmup=10, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def bench_mla_decode():
    results = []
    for B, H, N, L in [(1, 16, 1024, 512), (1, 32, 4096, 512), (4, 32, 8192, 512)]:
        Q = torch.randn(B, H, L, device="cuda", dtype=torch.float16)
        cKV = torch.randn(B, N, L, device="cuda", dtype=torch.float16)

        def ref():
            scale = 1.0 / (L**0.5)
            scores = torch.einsum("bhl,bnl->bhn", Q.float(), cKV.float()) * scale
            p = torch.softmax(scores, dim=-1)
            return torch.einsum("bhn,bnl->bhl", p, cKV.float())

        t_ours = _time(mla_decode, Q, cKV)
        t_ref = _time(ref)
        # memory bound: read cKV once (B*N*L*2 bytes)
        bytes_moved = B * N * L * 2
        results.append(
            {
                "B": B,
                "H": H,
                "N": N,
                "L": L,
                "ours_ms": t_ours * 1000,
                "ref_ms": t_ref * 1000,
                "speedup": t_ref / t_ours,
                "ours_gbs": bytes_moved / t_ours / 1e9,
            }
        )
        print(
            f"mla_decode B={B} H={H:2d} N={N:5d} L={L:3d} ours={t_ours*1000:6.2f}ms ref={t_ref*1000:6.2f}ms speedup={t_ref/t_ours:.2f}x"
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/ops_mla_decode.json")
    args = parser.parse_args()

    results = bench_mla_decode()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
