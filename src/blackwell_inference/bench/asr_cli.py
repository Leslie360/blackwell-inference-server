"""CLI entry for Qwen3-ASR benchmark."""

import argparse

from ..asr.qwen3_asr import benchmark_qwen3_asr, save_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to Qwen3-ASR model")
    parser.add_argument("--audio", required=True, help="path to audio file")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--output", default="benchmarks/asr.json")
    args = parser.parse_args()

    result = benchmark_qwen3_asr(
        model_path=args.model,
        audio_path=args.audio,
        batch_size=args.batch_size,
        compile=args.compile,
        num_runs=args.num_runs,
    )
    save_result(result, args.output)
    print(result)


if __name__ == "__main__":
    main()
