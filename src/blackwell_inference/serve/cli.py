"""CLI entry for Blackwell Inference Server."""

import argparse

import uvicorn

from .models import registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-model", help="path to Qwen3-ASR model")
    parser.add_argument("--llm-model", help="path to text LLM (e.g. Qwen3-0.6B)")
    parser.add_argument("--lora-adapter", help="path to LoRA adapter for the LLM")
    parser.add_argument("--compile", action="store_true", help="enable torch.compile")
    parser.add_argument("--attention-backend", default="sdpa", choices=["sdpa"], help="attention backend for ASR")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.asr_model:
        registry.load_asr(args.asr_model, compile=args.compile, attention_backend=args.attention_backend)
    if args.llm_model:
        registry.load_llm(args.llm_model, compile=args.compile, lora_adapter=args.lora_adapter)

    uvicorn.run("blackwell_inference.serve.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
