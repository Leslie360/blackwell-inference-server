# 在 RTX 5070 Ti 上自建 OpenAI 兼容推理服务：从 torch.compile 到自定义 Attention Kernel

> 项目地址：[github.com/Leslie360/blackwell-inference-server](https://github.com/Leslie360/blackwell-inference-server)  
> 硬件：RTX 5070 Ti 16GB（SM120, Blackwell）· WSL2 · CUDA 13.0

## 为什么做这个项目

消费级 Blackwell（RTX 5070 Ti / 5090）用的是 SM120 架构。它很强，但生态很差：

- CUTLASS 官方的 Blackwell FMHA 示例只支持 sm100a，不针对 sm120
- FlashAttention 2.x 没有官方 sm120 支持
- vLLM 0.14.0 依赖旧版 torch，装上去直接把环境搞崩
- WSL2 下 ncu/nsys 因 performance counter 权限问题抓不到 kernel

我想验证一件事：**在一张消费级显卡上，能不能搭出一个接近工业级体验的推理服务？** 答案是可以，但需要自己动手补全工具链。

## 做了什么

Blackwell Inference Server 是一个 OpenAI 兼容的推理服务，包含：

- **API 层**：`/v1/audio/transcriptions`、`/v1/chat/completions`、`/v1/models`
- **模型层**：Qwen3-ASR-0.6B/1.7B、Qwen3-0.6B
- **优化层**：torch.compile、四种 attention 后端、n-gram 投机解码
- **观测层**：torch.profiler、benchmark dashboard、Docker 一键复现

## 关键优化与数据

### 1. torch.compile：先救活框架

Qwen3-ASR 的 `generate()` 会绕过外层 `forward` 直接调 `thinker.generate`，导致 `torch.compile` 看起来跑了其实没生效。深入 modeling 源码后，改为对 `thinker` 解码器和 `audio_tower` 分别编译：

- baseline：**0.671s / 22.4× 实时倍率**
- torch.compile：**0.266s / 56.6× 实时倍率**
- **提升 2.45×**

大 batch 下提升有限（371× → 381×），因为瓶颈转移到 GEMM/FlashAttention。

### 2. 四种 Attention 后端对比

| 后端 | 类型 | N=2048 TFLOPS | 备注 |
|------|------|--------------:|------|
| SDPA | cuDNN flash attention | 78.7 | 公共库基线 |
| Linear | Triton chunked linear attention | 67.0 | 长上下文优势 |
| KDA | raw CUDA causal attention | 1.6 | 教育版，无 tensor core |
| Mini-Attention | SM120 mma.sync kernel | 68.7 | 接近硬件峰值 |

raw CUDA KDA 只有 1.6 TFLOPS，而 Mini-Attention 达 68.7 TFLOPS，差距 42×。这说明**没有 tensor core / warp specialization / TMEM-TMA，手写 kernel 很难打过 cuDNN**。

### 3. n-gram 自投机解码

实现了一个 self-speculative（prompt lookup）解码：

1. 用当前上下文建 n-gram 表
2. 从最近一次 n-gram 匹配中取 gamma 个 draft token
3. 一次 forward 验证整个 draft
4. 接受最长匹配前缀，截断 KV cache

结果：

- 重复文本（20× repeat）：**85.3 → 301.2 tok/s，3.53×，接受率 98.6%**
- 通用创作 prompt：**0.94×，几乎无开销**

结论：投机解码是 workload-dependent 的优化，重复/结构化文本收益巨大，通用文本不应开。

## 踩过的坑

1. **vLLM 0.14.0 会降级 torch 并破坏环境**：装完后 `ptxas-blackwell` 找不到，只能回滚。
2. **torchao 量化在 SM120 上不成熟**：INT8 weight-only / W8A8 / FP8 全部因 kernel 回退或动态缩放开销而劣化。
3. **ncu/nsys 在 WSL2 上权限不足**：`ERR_NVGPUCTRPERM`，只能用 torch.profiler 补齐证据链。
4. **torch.compile 的“假生效”**：必须确认实际被编译的 module 真的是 decode 入口。

## 工程化

- `pip install -e .` 即可安装
- `blackwell-serve --asr-model ... --llm-model ...` 一键起服务
- 内置 web dashboard，可直接跑 benchmark 和上传音频
- pytest 覆盖 attention / spec correctness
- Dockerfile + docker-compose 已提供

## 下一步

- 接 EAGLE 投机解码（比 n-gram 更通用）
- 支持更多模型（Qwen3-1.7B、Qwen3-VL）
- Hugging Face Spaces 在线 demo

## 项目地址

[github.com/Leslie360/blackwell-inference-server](https://github.com/Leslie360/blackwell-inference-server)
