import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from blackwell_inference.spec.ngram import greedy_generate, speculative_generate

MODEL_PATH = "/home/qiaosir/projects_1/qwen3_asr_accel/models/Qwen3-0.6B"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_spec_correctness():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda"
    )
    prompt = "The quick brown fox jumps over the lazy dog. " * 3
    ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    base = greedy_generate(model, ids, tokenizer, max_new_tokens=32)
    spec = speculative_generate(
        model, ids, tokenizer, max_new_tokens=32, gamma=4, ngram=3
    )

    assert base.text == spec.text
    assert base.tokens == spec.tokens
