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

## Benchmark methodology

- Attention: measure latency across N, compute causal FLOPs ≈ 2·B·H·N²·D
- ASR: report single-request latency and ×realtime throughput across batch sizes
- All benchmarks output JSON for reproducibility

## Roadmap

See `ROADMAP.md`.
