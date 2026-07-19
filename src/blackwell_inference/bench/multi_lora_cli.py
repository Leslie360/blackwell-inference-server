"""CLI for multi-LoRA serving benchmark."""

import argparse

from ..lora.multi import benchmark_multi_lora


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapters", required=True, help="name=path,name2=path2")
    parser.add_argument(
        "--prompt", default="The quick brown fox jumps over the lazy dog. " * 10
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    adapters = {}
    for item in args.adapters.split(","):
        name, path = item.split("=", 1)
        adapters[name.strip()] = path.strip()

    results = benchmark_multi_lora(
        base_model_path=args.base_model,
        adapters=adapters,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        repeats=args.repeats,
    )

    for r in results:
        print(
            f"{r.scenario:15s} {r.latency_s:.3f}s  {r.tokens_per_s:6.1f} tok/s  {r.memory_gb:.2f} GB"
        )


if __name__ == "__main__":
    main()
