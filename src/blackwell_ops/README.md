# blackwell-ops — Custom Operator Library for Consumer Blackwell

Triton and CUDA operators for RTX 5070 Ti (SM120).

## Performance vs PyTorch reference

| Operator | Backend | Speedup | Note |
|----------|---------|--------:|------|
| RMSNorm | Triton | **3.15–8.50×** | |
| RoPE | Triton | **3.16–7.12×** | |
| INT8 weight-only GEMM | Triton | **0.49×** | no int8 tensor core |
| RMSNorm | CUDA | **3.13–7.44×** | |
| Fused residual+RMSNorm | CUDA | **4.75–6.49×** | |
| RoPE | CUDA | **1.63–6.45×** | large tensor 6.45× |
| SwiGLU | CUDA | **2.63–5.14×** | |
| KV cache INT8 quant/dequant | CUDA | — | 2× compression, ~0.01ms |
| LoRA delta (B@A) | CUDA simple | **1.14× / 0.24×** | r=16 fast, large GEMM slow |
| **LoRA delta (B@A)** | **CUDA tiled** | **vs CUTLASS: 0.57× / vs cuBLAS: 0.24×** | **fp16 GEMM，与 CUTLASS 有 1.7× 差距** |
| MLA decode | CUDA | **0.05–0.13×** | structural demo, global-memory streaming |

## CUTLASS / cuBLAS comparison (LoRA delta, fp16)

| Kernel | r=128, K=4096, N=4096 | r=128, K=4096, N=14336 |
|--------|----------------------:|-----------------------:|
| ours (CUDA tiled) | 12.8 TFLOPS | 13.1 TFLOPS |
| CUTLASS FP16 (default config) | 22.3 TFLOPS | 23.1 TFLOPS |
| cuBLAS (torch.matmul) | **54.5 TFLOPS** | **62.5 TFLOPS** |

Our tiled CUDA kernel reaches ~57% of default CUTLASS FP16 and ~24% of cuBLAS. The gap comes from missing tensor-core warpgroup scheduling and software pipelining; cuBLAS on Blackwell uses heavily tuned CUTLASS kernels.

## Why some kernels are slower

- **INT8 GEMM**: no int8 tensor core path on SM120; converts to FP32 for accumulation.
- **LoRA delta large GEMM**: simple tiled CUDA without tensor cores cannot beat cuBLAS.
- **MLA decode**: SM120 has only **48KB shared memory per block**, limiting tile size for L=512 to TILE_N=16; educational demonstration of weight-absorption structure, not a performance kernel.

These negative results match our earlier torchao findings: on SM120, naive W8A16 / non-tensor-core GEMM is not a free lunch.

## Files

- `kernels/` — Triton kernels (rmsnorm, rope, int8_gemm)
- `cuda/` — CUDA kernels (basic_ops, kv_quant, lora_ops, mla_decode)
- `tests/` — correctness tests
- `bench/` — benchmarks vs PyTorch reference

## Run

```bash
pytest src/blackwell_ops/tests/
python -m blackwell_ops.bench.basic_bench
python -m blackwell_ops.bench.cuda_basic_bench
python -m blackwell_ops.bench.int8_gemm_bench
python -m blackwell_ops.bench.kv_quant_bench
python -m blackwell_ops.bench.lora_ops_bench
python -m blackwell_ops.bench.mla_decode_bench
```
