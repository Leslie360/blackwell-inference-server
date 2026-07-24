import pytest
import torch

from blackwell_ops.cuda import lora_delta_tiled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_delta_tiled():
    r, K, N = 16, 128, 96
    A = torch.randn(r, K, device="cuda", dtype=torch.float16)
    B = torch.randn(N, r, device="cuda", dtype=torch.float16)
    scaling = 2.0
    delta = lora_delta_tiled(A, B, scaling)
    ref = (B.float() @ A.float()) * scaling
    err = (delta.float() - ref).abs().max().item()
    assert err < 0.02
