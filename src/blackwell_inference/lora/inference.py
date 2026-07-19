"""LoRA inference: load adapter, merge, fused, and benchmark."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoRABenchResult:
    mode: str
    latency_s: float
    tokens_per_s: float
    memory_gb: float


class LoRAModel:
    def __init__(self, base_model_path: str, adapter_path: str | None = None, device: str = "cuda"):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self._base = None
        self._peft = None
        self._merged = None

    def load_base(self):
        if self._base is None:
            self._base = AutoModelForCausalLM.from_pretrained(
                self.base_model_path, torch_dtype=torch.float16, device_map=self.device
            )
        return self._base

    def load_peft(self):
        if self._peft is None:
            if not self.adapter_path:
                raise ValueError("adapter_path required")
            from peft import PeftModel

            base = self.load_base()
            self._peft = PeftModel.from_pretrained(base, self.adapter_path)
        return self._peft

    def load_merged(self):
        if self._merged is None:
            # Use a separate base copy so merge_and_unload does not contaminate the fused path.
            from peft import PeftModel

            base_copy = AutoModelForCausalLM.from_pretrained(
                self.base_model_path, torch_dtype=torch.float16, device_map=self.device
            )
            peft_copy = PeftModel.from_pretrained(base_copy, self.adapter_path)
            self._merged = peft_copy.merge_and_unload()
        return self._merged

    def generate(self, prompt: str, mode: str = "base", max_new_tokens: int = 64) -> str:
        if mode == "base":
            model = self.load_base()
        elif mode == "fused":
            model = self.load_peft()
        elif mode == "merged":
            model = self.load_merged()
        else:
            raise ValueError(f"unknown mode: {mode}")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def benchmark_lora(
    base_model_path: str,
    adapter_path: str,
    prompt: str,
    max_new_tokens: int = 64,
    repeats: int = 5,
) -> list[LoRABenchResult]:
    lora = LoRAModel(base_model_path, adapter_path)
    results = []

    for mode in ["base", "merged", "fused"]:
        # warmup
        lora.generate(prompt, mode=mode, max_new_tokens=8)
        torch.cuda.synchronize()

        latencies = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            lora.generate(prompt, mode=mode, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latency = sum(latencies) / len(latencies)
        results.append(
            LoRABenchResult(
                mode=mode,
                latency_s=latency,
                tokens_per_s=max_new_tokens / latency,
                memory_gb=torch.cuda.max_memory_allocated() / 1e9,
            )
        )
        torch.cuda.reset_peak_memory_stats()

    return results
