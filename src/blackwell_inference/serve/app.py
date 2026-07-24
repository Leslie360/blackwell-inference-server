"""Blackwell Inference Server — OpenAI-compatible API for consumer Blackwell GPUs."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from ..asr.qwen3_asr import benchmark_qwen3_asr
from ..bench.attention_runner import benchmark_attention
from ..bench.profile_attention import profile_attention
from .models import registry
from .schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelInfo,
    ModelList,
    TranscriptionResponse,
)

app = FastAPI(
    title="Blackwell Inference Server",
    description="OpenAI-compatible inference optimized for consumer Blackwell (RTX 5070 Ti, SM120)",
    version="0.1.0",
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _time_op(fn, repeats: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Blackwell Inference Server</h1>")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "models": len(registry.loaded),
    }


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    models = [ModelInfo(id=m["id"]) for m in registry.list_models()]
    return ModelList(data=models)


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...), model: str = "qwen3-asr", language: str | None = None
):
    if registry.asr is None:
        raise HTTPException(status_code=503, detail="ASR model not loaded")
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename or "audio.wav").suffix
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = registry.transcribe(tmp_path, language=language)
        return TranscriptionResponse(text=text)
    finally:
        os.unlink(tmp_path)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if registry.llm is None:
        raise HTTPException(status_code=503, detail="LLM model not loaded")
    prompt = "\n".join(f"{m.role}: {m.content}" for m in request.messages)
    text, stats = registry.generate_text(
        prompt,
        max_new_tokens=request.max_tokens,
        use_spec=request.use_spec,
        gamma=request.gamma,
        ngram=request.ngram,
    )
    return ChatCompletionResponse(
        id="chatcmpl-blackwell",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
            )
        ],
        usage={
            "prompt_tokens": 0,
            "completion_tokens": stats["tokens"],
            "total_tokens": stats["tokens"],
        },
    )


@app.post("/v1/benchmark", response_model=BenchmarkResponse)
async def benchmark(request: BenchmarkRequest):
    if request.task == "attention":
        result = benchmark_attention(
            backend=request.backend,
            batch=request.batch_size,
            seq_len=request.seq_len,
            repeats=request.repeats,
        )
        return BenchmarkResponse(
            task="attention",
            backend=request.backend,
            latency_ms=result.latency_ms,
            throughput=result.tflops,
            details={
                "seq_len": result.seq_len,
                "head_dim": result.head_dim,
                "max_err": result.max_err,
            },
        )
    if request.task == "asr":
        if not request.model or not request.audio:
            raise HTTPException(
                status_code=400, detail="model and audio required for ASR benchmark"
            )
        result = benchmark_qwen3_asr(
            model_path=request.model,
            audio_path=request.audio,
            batch_size=request.batch_size,
            compile=request.backend == "compile",
        )
        return BenchmarkResponse(
            task="asr",
            backend=request.backend,
            latency_ms=result.latency_s * 1000,
            throughput=result.throughput_x_realtime,
            details={"batch_size": result.batch_size, "memory_gb": result.memory_gb},
        )
    if request.task == "ops":
        from blackwell_ops.cuda import rmsnorm as cuda_rmsnorm
        from blackwell_ops.cuda import rope as cuda_rope
        from blackwell_ops.cuda import swiglu as cuda_swiglu
        from blackwell_ops.kernels import precompute_cos_sin

        if request.backend == "rmsnorm":
            x = torch.randn(
                request.batch_size, request.seq_len, device="cuda", dtype=torch.float16
            )
            w = torch.randn(request.seq_len, device="cuda", dtype=torch.float16)
            t = _time_op(lambda: cuda_rmsnorm(x, w), repeats=request.repeats)
        elif request.backend == "rope":
            x = torch.randn(
                request.batch_size,
                8,
                request.seq_len,
                64,
                device="cuda",
                dtype=torch.float16,
            )
            cos, sin = precompute_cos_sin(request.seq_len, 64, device="cuda")
            t = _time_op(lambda: cuda_rope(x, cos, sin), repeats=request.repeats)
        elif request.backend == "swiglu":
            gate = torch.randn(
                request.batch_size, request.seq_len, device="cuda", dtype=torch.float16
            )
            up = torch.randn(
                request.batch_size, request.seq_len, device="cuda", dtype=torch.float16
            )
            t = _time_op(lambda: cuda_swiglu(gate, up), repeats=request.repeats)
        else:
            raise HTTPException(
                status_code=400, detail=f"unknown ops backend: {request.backend}"
            )
        return BenchmarkResponse(
            task="ops",
            backend=request.backend,
            latency_ms=t * 1000,
            throughput=0.0,
            details={"batch_size": request.batch_size, "seq_len": request.seq_len},
        )
    raise HTTPException(status_code=400, detail=f"unknown task: {request.task}")


@app.get("/v1/profile")
async def profile(backend: str = "sdpa", seq_len: int = 2048):
    try:
        result = profile_attention(backend=backend, seq_len=seq_len)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(result["data"])


@app.get("/v1/profiles")
async def list_profiles():
    bench_dir = Path(__file__).resolve().parent.parent.parent.parent / "benchmarks"
    profiles = sorted(bench_dir.glob("profile_summary_*.json"))
    return {"profiles": [p.name for p in profiles]}
