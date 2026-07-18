"""Qwen3-ASR inference wrapper and benchmark helpers."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ASRResult:
    model: str
    batch_size: int
    compile: bool
    load_time_s: float
    latency_s: float
    throughput_x_realtime: float
    memory_gb: float


def _memory_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9


def benchmark_qwen3_asr(
    model_path: str,
    audio_path: str,
    batch_size: int = 1,
    compile: bool = False,
    num_runs: int = 3,
    max_new_tokens: int = 256,
) -> ASRResult:
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as e:
        raise RuntimeError("qwen-asr is required: pip install qwen-asr") from e

    t0 = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        max_new_tokens=max_new_tokens,
    )
    load_time = time.perf_counter() - t0
    torch.cuda.synchronize()

    if compile:
        hf_model = model.model
        hf_model.thinker.forward = torch.compile(hf_model.thinker.forward, mode="max-autotune-no-cudagraphs", dynamic=True)
        hf_model.thinker.audio_tower.forward = torch.compile(
            hf_model.thinker.audio_tower.forward, mode="max-autotune-no-cudagraphs", dynamic=True
        )

    audios = [audio_path] * batch_size
    # warmup with same batch size
    model.transcribe(audio=audios, language=None, return_time_stamps=False)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        model.transcribe(audio=audios, language=None, return_time_stamps=False)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    import soundfile as sf

    wav, sr = sf.read(audio_path)
    audio_duration_s = len(wav) / sr
    mean_latency = float(np.mean(latencies))
    throughput = audio_duration_s * batch_size / mean_latency

    return ASRResult(
        model=model_path,
        batch_size=batch_size,
        compile=compile,
        load_time_s=load_time,
        latency_s=mean_latency,
        throughput_x_realtime=throughput,
        memory_gb=_memory_gb(),
    )


def save_result(result: ASRResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
