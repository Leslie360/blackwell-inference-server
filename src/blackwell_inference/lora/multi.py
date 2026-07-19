"""Multi-LoRA serving: one base model, multiple adapters, dynamic switching."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class MultiLoRABenchResult:
    scenario: str
    latency_s: float
    tokens_per_s: float
    memory_gb: float


class MultiLoRAModel:
    def __init__(self, base_model_path: str, device: str = "cuda"):
        self.base_model_path = base_model_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.float16, device_map=device
        )
        self._peft = None
        self._current_adapter: str | None = None

    def load_adapter(self, name: str, adapter_path: str):
        from peft import PeftModel

        if self._peft is None:
            self._peft = PeftModel.from_pretrained(
                self.base, adapter_path, adapter_name=name
            )
        else:
            self._peft.load_adapter(adapter_path, adapter_name=name)

    def set_adapter(self, name: str):
        if self._peft is None:
            raise RuntimeError("no adapters loaded")
        if self._current_adapter != name:
            self._peft.set_adapter(name)
            self._current_adapter = name

    def generate(
        self, prompt: str, adapter: str | None = None, max_new_tokens: int = 64
    ) -> str:
        if adapter is None:
            model = self.base
        else:
            self.set_adapter(adapter)
            model = self._peft

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        return self.tokenizer.decode(
            out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )


def benchmark_multi_lora(
    base_model_path: str,
    adapters: dict[str, str],
    prompt: str,
    max_new_tokens: int = 64,
    repeats: int = 3,
) -> list[MultiLoRABenchResult]:
    model = MultiLoRAModel(base_model_path)
    results = []

    def _time(fn, *args, **kwargs):
        for _ in range(2):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / repeats

    # baseline
    t = _time(model.generate, prompt, adapter=None, max_new_tokens=max_new_tokens)
    results.append(
        MultiLoRABenchResult(
            "base", t, max_new_tokens / t, torch.cuda.max_memory_allocated() / 1e9
        )
    )
    torch.cuda.reset_peak_memory_stats()

    # each adapter
    for name, path in adapters.items():
        model.load_adapter(name, path)
        t = _time(model.generate, prompt, adapter=name, max_new_tokens=max_new_tokens)
        results.append(
            MultiLoRABenchResult(
                f"adapter:{name}",
                t,
                max_new_tokens / t,
                torch.cuda.max_memory_allocated() / 1e9,
            )
        )
        torch.cuda.reset_peak_memory_stats()

    # switching scenario: alternate between adapters
    names = list(adapters.keys())
    if len(names) >= 2:

        def _switch():
            model.generate(prompt, adapter=names[0], max_new_tokens=max_new_tokens)
            model.generate(prompt, adapter=names[1], max_new_tokens=max_new_tokens)

        t = _time(_switch) / 2
        results.append(
            MultiLoRABenchResult(
                "switching",
                t,
                max_new_tokens / t,
                torch.cuda.max_memory_allocated() / 1e9,
            )
        )

    return results
