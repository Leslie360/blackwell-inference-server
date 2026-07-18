"""Model registry for the inference server."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ASRModelHandle:
    model: Any
    compile: bool = False
    attention_backend: str = "sdpa"


@dataclass
class ModelRegistry:
    asr: ASRModelHandle | None = None
    llm: Any = None  # reserved for future text LLM
    loaded: dict[str, dict[str, Any]] = field(default_factory=dict)

    def load_asr(self, model_path: str, compile: bool = False, attention_backend: str = "sdpa") -> ASRModelHandle:
        if self.asr is not None:
            return self.asr
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as e:
            raise RuntimeError("qwen-asr is required") from e

        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation=attention_backend,
        )
        if compile:
            hf_model = model.model
            hf_model.thinker.forward = torch.compile(
                hf_model.thinker.forward, mode="max-autotune-no-cudagraphs", dynamic=True
            )
            hf_model.thinker.audio_tower.forward = torch.compile(
                hf_model.thinker.audio_tower.forward, mode="max-autotune-no-cudagraphs", dynamic=True
            )
        self.asr = ASRModelHandle(model=model, compile=compile, attention_backend=attention_backend)
        self.loaded["asr"] = {
            "path": model_path,
            "compile": compile,
            "attention_backend": attention_backend,
        }
        return self.asr

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        if self.asr is None:
            raise RuntimeError("ASR model not loaded")
        results = self.asr.model.transcribe(audio=audio_path, language=language, return_time_stamps=False)
        return results[0].text

    def list_models(self) -> list[dict[str, Any]]:
        models = []
        if self.asr is not None:
            models.append({"id": "qwen3-asr", "type": "asr", **self.loaded.get("asr", {})})
        if self.llm is not None:
            models.append({"id": "text-llm", "type": "llm"})
        return models


registry = ModelRegistry()
