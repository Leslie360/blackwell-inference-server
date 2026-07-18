"""CLI entry for attention benchmark."""

import argparse

from .attention_runner import run_grid, save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="sdpa,linear,kda,mini", help="comma-separated")
    parser.add_argument("--seq-lens", default="512,1024,2048,4096,8192", help="comma-separated")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", default="benchmarks/attention.json")
    args = parser.parse_args()

    backends = args.backends.split(",")
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    results = run_grid(backends, seq_lens, batch=args.batch, heads=args.heads, causal=args.causal, repeats=args.repeats)
    save_results(results, args.output)
    print(f"saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
