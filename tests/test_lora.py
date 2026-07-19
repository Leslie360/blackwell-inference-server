import pytest
import torch

from blackwell_inference.lora.inference import LoRAModel

BASE = "/home/qiaosir/projects_1/qwen3_asr_accel/models/Qwen3-0.6B"
ADAPTER = "/home/qiaosir/projects_1/blackwell_inference_kit/models/lora_qwen3_0.6b"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_merged_vs_fused():
    lora = LoRAModel(BASE, ADAPTER)
    prompt = "Hello, my name is"
    out_merged = lora.generate(prompt, mode="merged", max_new_tokens=16)
    out_fused = lora.generate(prompt, mode="fused", max_new_tokens=16)
    assert out_merged == out_fused
