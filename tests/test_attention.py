import math

import pytest
import torch

from blackwell_inference.attention import forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["sdpa", "linear", "kda"])
def test_attention_correctness(backend):
    B, H, N, D = 1, 4, 128, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) / math.sqrt(D)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) / math.sqrt(D)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    out = forward(backend, q, k, v, causal=True)
    if backend == "linear":
        # linear attention = unnormalized causal sum, not softmax
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
        mask = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, 0.0)
        ref = torch.matmul(scores, v.float()).to(q.dtype)
    else:
        ref = forward("sdpa", q, k, v, causal=True)
    atol = 0.05 * ref.abs().max().item()
    err = (out.float() - ref.float()).abs().max().item()
    assert err < atol


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mini_backend_optional():
    try:
        from blackwell_inference.attention.mini import forward as mini_forward
    except RuntimeError:
        pytest.skip("Mini-Attention not built")
    B, H, N, D = 1, 4, 128, 128
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) / math.sqrt(D)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) / math.sqrt(D)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    out = mini_forward(q, k, v)
    assert out.shape == q.shape
