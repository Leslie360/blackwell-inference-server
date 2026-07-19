import pytest
import torch

from blackwell_ops.kernels import precompute_cos_sin, rmsnorm, rope


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rmsnorm():
    M, N = 64, 512
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    w = torch.randn(N, device="cuda", dtype=torch.float16)

    out = rmsnorm(x, w)
    # reference
    x_f = x.float()
    ref = x_f / torch.sqrt((x_f * x_f).mean(dim=-1, keepdim=True) + 1e-6) * w.float()
    err = (out.float() - ref).abs().max().item()
    assert err < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rope():
    B, H, N, D = 2, 4, 128, 64
    x = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    cos, sin = precompute_cos_sin(N, D, device="cuda")

    out = rope(x, cos, sin)
    # reference
    x1 = x[..., : D // 2].float()
    x2 = x[..., D // 2 :].float()
    ref1 = x1 * cos - x2 * sin
    ref2 = x1 * sin + x2 * cos
    ref = torch.cat([ref1, ref2], dim=-1)
    err = (out.float() - ref).abs().max().item()
    assert err < 0.01
