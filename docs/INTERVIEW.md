# 面试 Talking Points — Blackwell Inference Server

## 一句话介绍

“我在 RTX 5070 Ti 上独立做了一个 OpenAI 兼容推理服务，集成了自定义 attention kernel、torch.compile 优化和 n-gram 投机解码，Qwen3-ASR 单请求吞吐提升 2.45×，重复文本生成 3.53×，GitHub 可一键复现。”

## 可能被问的问题与回答

### Q1：为什么不用 vLLM / SGLang？

A：消费级 SM120 支持不好。vLLM 0.14.0 依赖旧版 torch，装上后会破坏 Blackwell 的 ptxas；CUTLASS Blackwell FMHA 只支持 sm100a；FlashAttention 2.x 没有官方 sm120 支持。所以我在消费级卡上选择自研服务，但保留了和 SDPA/cuDNN 的对比。

### Q2：2.45× 是怎么来的？

A：主要是 torch.compile。Qwen3-ASR 的 `generate()` 会绕过外层 forward 直接调 `thinker.generate`，导致 `torch.compile` 失效。我编译了 thinker 解码器和 audio_tower，单请求延迟从 0.671s 降到 0.266s。大 batch 下提升有限，因为瓶颈转移到 GEMM/attention。

### Q3：为什么自研 KDA kernel 比 cuDNN 慢 40×？

A：KDA 是教学版 raw CUDA，没有 tensor core、没有 warp specialization、没有 TMEM/TMA。Mini-Attention 用 mma.sync + swizzling + double buffering 能达到 87.5 TFLOPS，说明差距主要在硬件特性利用上。这个对比帮我理解了工业化 kernel 的复杂度。

### Q4：n-gram 投机解码的原理？

A：self-speculative / prompt lookup。用当前上下文建 n-gram 表，取最近一次匹配后的 gamma 个 token 作为 draft，一次 forward 验证整个 draft，接受最长匹配前缀并截断 KV cache。重复文本接受率 98.6%，通用文本几乎无开销。

### Q5：为什么通用文本不加速？

A：投机解码依赖输出中的重复模式。创作类文本几乎没有可复用的 n-gram，draft 接受率只有 4%，验证开销抵消了收益。所以我在 API 里加了 `use_spec` 开关，默认关闭。

### Q6：量化为什么没做？

A：系统评估了 torchao INT8 weight-only / W8A8 / FP8，在 SM120 上都因 kernel 回退或动态缩放开销而劣化。用数据说明“量化不一定更快”，这是我特意做的负面验证。

### Q7：ncu/nsys 为什么没用？

A：WSL2 默认禁止用户态访问 GPU performance counters（`ERR_NVGPUCTRPERM`），所以 kernel 级分析用 torch.profiler。在原生 Linux 下可以直接换 ncu/nsys。

### Q8：如果让你继续做，下一步是什么？

A：接 EAGLE 投机解码（用一个小 draft model 替代 n-gram，通用文本也能加速）；支持更多模型；做 Hugging Face Spaces 在线 demo。

## 关键数字速查

| 指标 | 数值 |
|------|------|
| Qwen3-ASR baseline | 0.671s / 22.4× |
| Qwen3-ASR + torch.compile | 0.266s / 56.6×（2.45×） |
| Mini-Attention 8k | 87.5 TFLOPS（93% 硬件峰值） |
| n-gram spec 重复文本 | 3.53× / 98.6% 接受率 |
| n-gram spec 通用文本 | 0.94× / 4.2% 接受率 |
| raw CUDA KDA | 1.6 TFLOPS（vs cuDNN 42× gap） |
