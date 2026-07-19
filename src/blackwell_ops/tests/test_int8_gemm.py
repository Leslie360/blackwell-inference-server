import pytest
import torch

from blackwell_ops.kernels import int8_gemm, quantize_weight_int8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int8_gemm():
    M, K, N = 64, 128, 96
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16)

    w_int8, scale = quantize_weight_int8(w)
    out = int8_gemm(x, w_int8, scale)

    # reference: dequantize then matmul
    w_deq = w_int8.float() * scale.unsqueeze(-1)
    ref = x.float() @ w_deq.t()
    err = (out.float() - ref).abs().max().item()
    # int8 quantization introduces error, allow larger tolerance
    assert err < 0.5
