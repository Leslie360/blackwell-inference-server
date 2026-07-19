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
