"""CLI for n-gram speculative decoding benchmark."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..spec.ngram import greedy_generate, speculative_generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. " * 20)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--ngram", type=int, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    inputs = tokenizer(args.prompt, return_tensors="pt").input_ids.cuda()

    # warmup
    _ = greedy_generate(model, inputs, tokenizer, max_new_tokens=8)
    torch.cuda.synchronize()

    base = greedy_generate(model, inputs, tokenizer, max_new_tokens=args.max_new_tokens)
    spec = speculative_generate(model, inputs, tokenizer, max_new_tokens=args.max_new_tokens,
                                gamma=args.gamma, ngram=args.ngram)

    print(f"baseline: {base.tokens_per_s:.1f} tok/s ({base.wall_time_s:.3f}s)")
    print(f"speculative: {spec.tokens_per_s:.1f} tok/s ({spec.wall_time_s:.3f}s) "
          f"accept_rate={spec.draft_acceptance_rate:.2%}")
    print(f"speedup: {spec.tokens_per_s / base.tokens_per_s:.2f}x")
    print(f"same text: {base.text == spec.text}")


if __name__ == "__main__":
    main()
