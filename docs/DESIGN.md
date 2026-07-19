# Design

## Why consumer Blackwell (SM120) needs its own toolkit

Consumer Blackwell (RTX 5070 Ti / 5090) uses the SM120 architecture. It has tensor cores and supports `mma.sync`, but lacks Hopper's TMEM/WGMMA and many server-side Blackwell features (sm100a). As a result:

- CUTLASS official Blackwell FMHA targets sm100a, not sm120
- FlashAttention 2.x does not officially support sm120
- torchao INT8/FP8 kernels are immature on sm120
- WSL2 blocks ncu/nsys kernel profiling

This project packages working attention kernels, ASR acceleration recipes, and benchmark harnesses specifically for sm120.

## Attention backends

| Backend | Type | Head dim | Causal | Notes |
|---------|------|---------:|--------|-------|
| `sdpa` | PyTorch SDPA / cuDNN | any | both | public-library baseline |
| `linear` | Triton chunked linear attention | 64/128 | yes | O(N·D²), long-context friendly |
| `kda` | raw CUDA fused causal attention | 64 | yes | educational, shows tensor-core gap |
| `mini` | Mini-Attention SM120 | 128 | no | high-performance mma.sync kernel |

## ASR acceleration

Qwen3-ASR uses a `thinker` decoder + `audio_tower`. `torch.compile` on these two modules reduces Python generate-loop overhead and fuses small kernels, yielding ~2.45× single-request speedup on RTX 5070 Ti.

## LoRA inference

`blackwell_inference.lora` supports:

- loading PEFT LoRA adapters for Qwen3-0.6B
- weight merge (`merge_and_unload`) vs fused inference (no merge)
- correctness test (merged and fused produce identical output)
- benchmark CLI (`blackwell-lora-bench`) and multi-LoRA CLI (`blackwell-multi-lora-bench`)

**Corrected findings (after fixing a contamination bug)**:

| Mode | r=16 tok/s | r=128 tok/s |
|------|-----------:|------------:|
| base | 78.5 | 77.4 |
| merged | 78.9 | 77.8 |
| fused | 52.6 | 53.8 |

- merged ≈ base：LoRA 合权不伤性能
- fused 慢 ~33%：额外 LoRA GEMM 开销
- 多 LoRA serving 用 fused 换显存节省

**Bug**: 早期版本 fused 显示和 base 一样快，是因为 `merge_and_unload` 原地修改了共享 base model，污染了 fused 路径。修复方法：merge 到独立副本。

## Speculative decoding

`blackwell_inference.spec.ngram` implements self-speculative (n-gram / prompt-lookup) decoding:

1. Build an n-gram table from the current context
2. Propose `gamma` draft tokens from the most recent n-gram match
3. Verify the draft with the target model in one forward pass
4. Accept the longest matching prefix and truncate KV cache

On highly repetitive text (20× repeat), it reaches **3.53× speedup with 98.6% acceptance**; on general text it adds negligible overhead. This is honest, measurable evidence that speculative decoding is a workload-dependent optimization.

## Benchmark methodology

- Attention: measure latency across N, compute causal FLOPs ≈ 2·B·H·N²·D
- ASR: report single-request latency and ×realtime throughput across batch sizes
- All benchmarks output JSON for reproducibility

## Roadmap

See `ROADMAP.md`.
