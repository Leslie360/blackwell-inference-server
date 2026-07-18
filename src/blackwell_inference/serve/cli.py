"""CLI entry for Blackwell Inference Server."""

import argparse

import uvicorn

from .models import registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-model", help="path to Qwen3-ASR model")
    parser.add_argument("--compile", action="store_true", help="enable torch.compile for ASR")
    parser.add_argument("--attention-backend", default="sdpa", choices=["sdpa"], help="attention backend for ASR")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.asr_model:
        registry.load_asr(args.asr_model, compile=args.compile, attention_backend=args.attention_backend)

    uvicorn.run("blackwell_inference.serve.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
