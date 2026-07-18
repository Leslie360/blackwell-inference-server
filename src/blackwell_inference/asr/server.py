"""Minimal FastAPI server for Qwen3-ASR."""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
from fastapi import FastAPI, File, UploadFile

from ..asr.qwen3_asr import benchmark_qwen3_asr

app = FastAPI(title="blackwell-inference-kit ASR")

_model = None
_compile = False


def _load_model(model_path: str, compile: bool):
    global _model, _compile
    if _model is not None:
        return _model
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as e:
        raise RuntimeError("qwen-asr is required") from e

    _model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    if compile:
        hf_model = _model.model
        hf_model.thinker.forward = torch.compile(hf_model.thinker.forward, mode="max-autotune-no-cudagraphs", dynamic=True)
        hf_model.thinker.audio_tower.forward = torch.compile(
            hf_model.thinker.audio_tower.forward, mode="max-autotune-no-cudagraphs", dynamic=True
        )
    _compile = compile
    return _model


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        results = _model.transcribe(audio=tmp_path, language=None, return_time_stamps=False)
        return {"text": results[0].text, "compile": _compile}
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    _load_model(args.model, args.compile)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
