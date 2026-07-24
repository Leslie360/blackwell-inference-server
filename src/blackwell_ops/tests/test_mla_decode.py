import pytest
import torch

from blackwell_ops.cuda import mla_decode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mla_decode():
    B, H, N, L = 2, 4, 128, 64
    Q = torch.randn(B, H, L, device="cuda", dtype=torch.float16)
    cKV = torch.randn(B, N, L, device="cuda", dtype=torch.float16)

    out = mla_decode(Q, cKV)

    # reference: standard softmax attention over latent cache
    scale = 1.0 / (L**0.5)
    scores = torch.einsum("bhl,bnl->bhn", Q.float(), cKV.float()) * scale
    p = torch.softmax(scores, dim=-1)
    ref = torch.einsum("bhn,bnl->bhl", p, cKV.float())

    err = (out.float() - ref).abs().max().item()
    assert err < 0.02
