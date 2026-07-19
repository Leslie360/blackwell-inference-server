"""LoRA delta precomputation: fuse B@A into base weight for single-GEMM inference."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_lora_delta(peft_model) -> dict[str, torch.Tensor]:
    """Extract per-layer LoRA delta weights (B @ A * scaling) from a PEFT model."""
    deltas = {}
    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            # peft LoRA Linear stores adapters in ModuleDict keyed by adapter name
            for adapter_name in module.lora_A.keys():
                A = module.lora_A[adapter_name].weight  # [r, d_in]
                B = module.lora_B[adapter_name].weight  # [d_out, r]
                scaling = module.scaling[adapter_name]
                delta = (B @ A) * scaling  # [d_out, d_in]
                key = f"{name}.{adapter_name}"
                deltas[key] = delta
    return deltas


def apply_lora_delta(model, deltas: dict[str, torch.Tensor], adapter_name: str = "default"):
    """Apply precomputed LoRA delta weights to the base model in-place.

    Replaces each LoRA Linear with an equivalent standard Linear whose weight is
    W0 + delta. Returns a list of (module, original_weight) for optional rollback.
    """
    backups = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and adapter_name in module.lora_A:
            key = f"{name}.{adapter_name}"
            if key not in deltas:
                continue
            base_layer = module.base_layer
            delta = deltas[key].to(base_layer.weight.device, base_layer.weight.dtype)
            # backup original weight
            backups.append((base_layer, base_layer.weight.data.clone()))
            # fuse delta into base weight
            base_layer.weight.data += delta
    return backups


def rollback_lora_delta(backups):
    for base_layer, original_weight in backups:
        base_layer.weight.data = original_weight


class DeltaLoRAModel:
    """LoRA inference with precomputed delta weights (single GEMM per layer)."""

    def __init__(self, base_model_path: str, adapter_path: str, device: str = "cuda"):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.float16, device_map=device
        )
        self._peft = None
        self._backups = None

    def load_peft(self):
        if self._peft is None:
            from peft import PeftModel

            self._peft = PeftModel.from_pretrained(self.base, self.adapter_path)
        return self._peft

    def prepare_delta(self):
        """Extract delta and fuse into base weights."""
        peft_model = self.load_peft()
        deltas = extract_lora_delta(peft_model)
        self._backups = apply_lora_delta(self.base, deltas)

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        if self._backups is None:
            self.prepare_delta()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.base.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
