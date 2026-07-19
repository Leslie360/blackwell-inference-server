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
class LLMModelHandle:
    model: Any
    tokenizer: Any
    compile: bool = False


@dataclass
class ModelRegistry:
    asr: ASRModelHandle | None = None
    llm: LLMModelHandle | None = None
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

    def load_llm(self, model_path: str, compile: bool = False, lora_adapter: str | None = None) -> LLMModelHandle:
        if self.llm is not None:
            return self.llm
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="cuda",
        )
        if lora_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, lora_adapter)
        if compile:
            model.forward = torch.compile(model.forward, mode="max-autotune-no-cudagraphs", dynamic=True)
        self.llm = LLMModelHandle(model=model, tokenizer=tokenizer, compile=compile)
        self.loaded["llm"] = {"path": model_path, "compile": compile, "lora_adapter": lora_adapter}
        return self.llm

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        if self.asr is None:
            raise RuntimeError("ASR model not loaded")
        results = self.asr.model.transcribe(audio=audio_path, language=language, return_time_stamps=False)
        return results[0].text

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        use_spec: bool = False,
        gamma: int = 4,
        ngram: int = 3,
    ) -> tuple[str, dict[str, Any]]:
        if self.llm is None:
            raise RuntimeError("LLM model not loaded")
        from ..spec.ngram import greedy_generate, speculative_generate

        inputs = self.llm.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        if use_spec:
            result = speculative_generate(
                self.llm.model, inputs, self.llm.tokenizer, max_new_tokens=max_new_tokens, gamma=gamma, ngram=ngram
            )
        else:
            result = greedy_generate(self.llm.model, inputs, self.llm.tokenizer, max_new_tokens=max_new_tokens)
        return result.text, {
            "tokens": result.tokens,
            "wall_time_s": result.wall_time_s,
            "tokens_per_s": result.tokens_per_s,
            "draft_acceptance_rate": result.draft_acceptance_rate,
        }

    def list_models(self) -> list[dict[str, Any]]:
        models = []
        if self.asr is not None:
            models.append({"id": "qwen3-asr", "type": "asr", **self.loaded.get("asr", {})})
        if self.llm is not None:
            models.append({"id": "qwen3-text", "type": "llm", **self.loaded.get("llm", {})})
        return models


registry = ModelRegistry()
