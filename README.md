# blackwell-inference-kit

Inference optimization toolkit for consumer Blackwell GPUs (RTX 5070 Ti, SM120): custom attention kernels, Qwen3-ASR acceleration, and reproducible benchmarks.

## What it does

- **Attention backends**: PyTorch SDPA, Triton linear attention, raw CUDA causal attention (KDA), and optional Mini-Attention SM120 kernel
- **ASR acceleration**: Qwen3-ASR inference with baseline / torch.compile comparison
- **Benchmarks**: one-command attention and ASR benchmarks with JSON output
- **HTTP server**: minimal FastAPI wrapper for Qwen3-ASR

## Quick start

```bash
git clone https://github.com/yourname/blackwell-inference-kit.git
cd blackwell-inference-kit

# 1. Install (editable)
pip install -e .[asr,server]

# 2. Attention benchmark (SDPAs, linear, KDA; mini optional)
blackwell-attn-bench --backends sdpa,linear,kda --seq-lens 512,1024,2048

# 3. Qwen3-ASR benchmark
blackwell-asr-bench \
  --model /path/to/Qwen3-ASR-0.6B \
  --audio /path/to/audio.wav \
  --batch-size 1

# 4. Qwen3-ASR with torch.compile
blackwell-asr-bench \
  --model /path/to/Qwen3-ASR-0.6B \
  --audio /path/to/audio.wav \
  --batch-size 1 --compile

# 5. HTTP server
blackwell-serve --model /path/to/Qwen3-ASR-0.6B --compile --port 8000
```

## Optional: Mini-Attention backend

The `mini` backend uses the open-source [ShlokVFX/Mini-Attention](https://github.com/ShlokVFX/Mini-Attention) SM120 kernel. Clone it and set:

```bash
export MINI_ATTENTION_ROOT=/path/to/Mini-Attention
```

Build the extension first following its README, then run:

```bash
blackwell-attn-bench --backends sdpa,mini --seq-lens 4096,8192
```

## Benchmark results (RTX 5070 Ti)

See `benchmarks/` for raw JSON. Representative numbers:

| Backend | N | TFLOPS | Note |
|---------|---|-------:|------|
| SDPA | 2048 | 78.7 | cuDNN flash attention |
| linear | 2048 | 67.0 | Triton chunked linear attention |
| KDA | 2048 | 1.6 | raw CUDA educational kernel |
| mini | 2048 | 68.7 | SM120 mma.sync kernel |

Qwen3-ASR-0.6B:

| Config | Latency | Throughput |
|--------|--------:|-----------:|
| baseline | 0.671 s | 22.4× realtime |
| torch.compile | 0.266 s | **56.6× realtime** |

## Requirements

- CUDA 13.0 / PyTorch >= 2.8 (compiled for SM120)
- `nvcc` from `nvidia/cu13` PyPI package or system CUDA >= 12.9
- Optional: `qwen-asr`, `modelscope`, `fastapi`, `uvicorn`

## Project layout

```
src/blackwell_inference/
├── attention/
│   ├── sdpa backend (torch)
│   ├── linear/      # Triton linear attention
│   ├── kda/         # raw CUDA causal attention
│   └── mini/        # Mini-Attention wrapper
├── asr/             # Qwen3-ASR wrapper + FastAPI server
├── bench/           # benchmark runners + CLIs
├── spec/            # speculative decoding (placeholder)
└── utils/
tests/
benchmarks/
docs/
```

## License

MIT
