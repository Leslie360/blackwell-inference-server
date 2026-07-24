# Roadmap

## v0.1 (MVP — current)

- [x] Package skeleton, MIT license, pyproject
- [x] Attention backends: SDPA, linear, KDA, mini (optional)
- [x] Qwen3-ASR baseline + torch.compile benchmark
- [x] Attention/ASR benchmark CLIs
- [x] FastAPI ASR server
- [x] pytest correctness tests
- [x] n-gram speculative decoding with correctness + benchmark
- [x] LoRA adapter loading / merge / fused inference with correctness + benchmark
- [x] multi-LoRA serving benchmark (fused overhead ~33%)
- [x] **delta-fused LoRA optimization** (precompute ΔW=B@A, eliminate fused overhead)
- [x] **blackwell-ops operator library**: RMSNorm, RoPE, INT8 GEMM (correctness + benchmark)
- [x] **blackwell-ops CUDA kernels**: RMSNorm, fused residual+RMSNorm, RoPE, SwiGLU, KV INT8 quant, LoRA delta
- [x] **tiled LoRA delta kernel**: shared-memory tiling; ~57% of default CUTLASS FP16, ~24% of cuBLAS
- [x] **MLA decode kernel**: weight-absorption structure, global-memory streaming
- [x] **CUTLASS comparison**: measured CUTLASS FP16 vs our kernel vs cuBLAS
- [ ] README + docs complete

## v0.2

- [ ] GitHub Actions CI (build + test)
- [ ] PyPI source distribution
- [ ] Benchmark dashboard (GitHub Pages / MkDocs)
- [ ] More Qwen3-ASR batch/quant experiments
- [ ] Speculative decoding components (EAGLE / n-gram)

## v1.0

- [ ] SGLang / vLLM integration examples
- [ ] Hugging Face Spaces demo
- [ ] Blog post + video walkthrough
- [ ] Stable API and semantic versioning
