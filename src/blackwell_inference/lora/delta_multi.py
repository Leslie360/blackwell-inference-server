"""Multi-LoRA serving with precomputed delta weights and zero-GEMM-overhead switching."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DeltaMultiBenchResult:
    scenario: str
    latency_s: float
    tokens_per_s: float
    memory_gb: float


class MultiDeltaLoRAModel:
    """One base model + per-adapter precomputed deltas fused into weights."""

    def __init__(self, base_model_path: str, device: str = "cuda"):
        self.base_model_path = base_model_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.float16, device_map=device
        )
        self._lora_layers: dict[str, torch.nn.Linear] = {}
        self._deltas: dict[str, dict[str, torch.Tensor]] = {}
        self._current_adapter: str | None = None
        self._collect_lora_layers()

    def _collect_lora_layers(self):
        for name, module in self.base.named_modules():
            if isinstance(module, torch.nn.Linear) and any(
                k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")
            ):
                self._lora_layers[name] = module

    def load_adapter(self, name: str, adapter_path: str):
        from peft import PeftModel

        # Use a temporary copy so the base model is not mutated by peft injection.
        tmp_base = AutoModelForCausalLM.from_pretrained(
            self.base_model_path, torch_dtype=torch.float16, device_map=self.device
        )
        peft = PeftModel.from_pretrained(tmp_base, adapter_path)
        deltas = {}
        for layer_name, linear in self._lora_layers.items():
            # peft.base_model is LoraModel; .model is Qwen3ForCausalLM
            module = peft.base_model.model.get_submodule(layer_name)
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                adapter_name = list(module.lora_A.keys())[0]
                A = module.lora_A[adapter_name].weight
                B = module.lora_B[adapter_name].weight
                scaling = module.scaling[adapter_name]
                delta = (B @ A) * scaling
                deltas[layer_name] = delta.to(linear.weight.device, linear.weight.dtype)
        self._deltas[name] = deltas
        del peft, tmp_base
        torch.cuda.empty_cache()

    def _apply_delta(self, adapter_name: str):
        if self._current_adapter == adapter_name:
            return
        # remove old delta
        if self._current_adapter is not None:
            for layer_name, delta in self._deltas[self._current_adapter].items():
                self._lora_layers[layer_name].weight.data -= delta
        # add new delta
        if adapter_name is not None:
            for layer_name, delta in self._deltas[adapter_name].items():
                self._lora_layers[layer_name].weight.data += delta
        self._current_adapter = adapter_name

    def generate(self, prompt: str, adapter: str | None = None, max_new_tokens: int = 64) -> str:
        self._apply_delta(adapter)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.base.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def benchmark_delta_multi(
    base_model_path: str,
    adapters: dict[str, str],
    prompt: str,
    max_new_tokens: int = 64,
    repeats: int = 3,
) -> list[DeltaMultiBenchResult]:
    model = MultiDeltaLoRAModel(base_model_path)
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
    results.append(DeltaMultiBenchResult("base", t, max_new_tokens / t, torch.cuda.max_memory_allocated() / 1e9))
    torch.cuda.reset_peak_memory_stats()

    # each adapter
    for name, path in adapters.items():
        model.load_adapter(name, path)
        t = _time(model.generate, prompt, adapter=name, max_new_tokens=max_new_tokens)
        results.append(
            DeltaMultiBenchResult(f"adapter:{name}", t, max_new_tokens / t, torch.cuda.max_memory_allocated() / 1e9)
        )
        torch.cuda.reset_peak_memory_stats()

    # switching scenario
    names = list(adapters.keys())
    if len(names) >= 2:

        def _switch():
            model.generate(prompt, adapter=names[0], max_new_tokens=max_new_tokens)
            model.generate(prompt, adapter=names[1], max_new_tokens=max_new_tokens)

        t = _time(_switch) / 2
        results.append(DeltaMultiBenchResult("switching", t, max_new_tokens / t, torch.cuda.max_memory_allocated() / 1e9))

    return results
