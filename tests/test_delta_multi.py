import pytest
import torch

from blackwell_inference.lora.delta_multi import MultiDeltaLoRAModel
from blackwell_inference.lora.inference import LoRAModel

BASE = "/home/qiaosir/projects_1/qwen3_asr_accel/models/Qwen3-0.6B"
ADAPTER = "/home/qiaosir/projects_1/blackwell_inference_kit/models/lora_qwen3_0.6b"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_delta_multi_correctness():
    prompt = "Hello, my name is"
    multi = MultiDeltaLoRAModel(BASE)
    multi.load_adapter("r16", ADAPTER)
    out_delta = multi.generate(prompt, adapter="r16", max_new_tokens=16)

    lora = LoRAModel(BASE, ADAPTER)
    out_fused = lora.generate(prompt, mode="fused", max_new_tokens=16)

    assert out_delta == out_fused
