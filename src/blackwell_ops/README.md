# blackwell-ops — Custom Operator Library for Consumer Blackwell

Triton/CUDA operators optimized for RTX 5070 Ti (SM120).

## Operators

| Operator | Type | Correctness | Performance vs PyTorch |
|----------|------|-------------|------------------------|
| RMSNorm | Triton | ✅ | **3.15–8.50× faster** |
| RoPE | Triton | ✅ | **3.16–7.12× faster** |
| INT8 weight-only GEMM | Triton | ✅ | **0.49× slower** (no int8 tensor core) |

## Why INT8 GEMM is slower

Our INT8 kernel loads int8 weights but converts them to FP32 for accumulation; it does not use SM120's int8 tensor cores (`mma.sync.s8`). On this card, a naive W8A16 GEMM cannot beat cuBLAS FP16. This matches our earlier finding with torchao INT8 weight-only.

## Files

- `kernels/rmsnorm.py` — RMSNorm forward
- `kernels/rope.py` — RoPE with precomputed cos/sin
- `kernels/int8_gemm.py` — INT8 weight-only GEMM
- `tests/` — correctness tests
- `bench/` — benchmarks vs PyTorch reference

## Run

```bash
pytest src/blackwell_ops/tests/
python -m blackwell_ops.bench.basic_bench
python -m blackwell_ops.bench.int8_gemm_bench
```
