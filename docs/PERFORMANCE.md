# 性能分析（ncu/nsys 与 torch.profiler）

## ncu / nsys 状态

在 **WSL2 + RTX 5070 Ti (SM120)** 环境下，`ncu`/`nsys` 无法采集 CUDA kernel trace：

- `ncu` 未安装（系统无此命令）
- `nsys profile` 能生成报告，但 `nsys stats` 显示：
  - `SKIPPED: ... does not contain CUDA kernel data`
  - `SKIPPED: ... does not contain CUDA trace data`
  - 只有 OS runtime（`poll`/`read`/`ioctl`）统计

因此本项目的 kernel 级性能分析全部依赖 **torch.profiler**（Chrome trace + key averages），这是当前环境下唯一可靠的手段。

## 已采集的 profile

使用 `blackwell_inference.bench.profile_attention` 对 N=2048, B=1, H=8 采集了四种后端的 trace：

| 后端 | 单次 kernel 时间 | top kernel | 说明 |
|------|-----------------:|------------|------|
| SDPA | ~0.098 ms | `aten::_flash_attention_forward` | cuDNN flash attention |
| linear | ~0.020 ms | `_chunk_fwd_o_kernel` | Triton chunked linear attention |
| KDA | ~2.50 ms | `kda_fwd_kernel<c10::Half>` | raw CUDA，无 tensor core |
| mini | ~0.249 ms | `flash::flash_forward_kernel` | SM120 mma.sync kernel |

原始文件在 `benchmarks/profile_*_n2048.json`。

## 性能差距的根因

1. **SDPA / mini 接近硬件极限**：它们使用 tensor core（WGMMA/mma.sync）、融合 epilogue 和高效 shared-memory 调度，N=2048 达到 ~70–80 TFLOPS。
2. **KDA raw CUDA 慢 40×**：没有 tensor core、没有 warp specialization、没有 double buffering，scalar FMA 吞吐被完全压制。
3. **linear attention 极快**：复杂度 O(N·D²) 而不是 O(N²D)，在长上下文下优势会持续放大。

## 复现

```bash
# attention profile
python -m blackwell_inference.bench.profile_attention --backend sdpa --seq-len 2048

# 查看 top kernel
cat benchmarks/profile_summary_sdpa_n2048.json
```
