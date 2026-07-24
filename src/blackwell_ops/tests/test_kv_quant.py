import pytest
import torch

from blackwell_ops.cuda import dequantize_kv_int8, quantize_kv_int8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kv_quant_roundtrip():
    C, N = 32, 4096
    x = torch.randn(C, N, device="cuda", dtype=torch.float16)
    q, scale = quantize_kv_int8(x)
    x_deq = dequantize_kv_int8(q, scale)
    err = (x.float() - x_deq.float()).abs().max().item()
    # int8 quantization error
    assert err < 0.1
    assert q.dtype == torch.int8
    assert scale.dtype == torch.float32
