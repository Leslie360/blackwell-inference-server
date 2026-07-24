import pytest
import torch

from blackwell_ops.cuda import add_delta_, lora_delta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_delta():
    r, K, N = 16, 128, 96
    A = torch.randn(r, K, device="cuda", dtype=torch.float16)
    B = torch.randn(N, r, device="cuda", dtype=torch.float16)
    scaling = 2.0
    delta = lora_delta(A, B, scaling)
    ref = (B.float() @ A.float()) * scaling
    err = (delta.float() - ref).abs().max().item()
    assert err < 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_add_delta():
    N, K = 64, 128
    W = torch.randn(N, K, device="cuda", dtype=torch.float16)
    delta = torch.randn(N, K, device="cuda", dtype=torch.float16)
    W_orig = W.clone()
    add_delta_(W, delta)
    ref = W_orig.float() + delta.float()
    err = (W.float() - ref).abs().max().item()
    assert err < 0.01
