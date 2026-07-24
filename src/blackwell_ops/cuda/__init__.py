"""CUDA operator library for blackwell-ops (lazy-built)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_ext = None


def _ensure_nvcc():
    import torch

    cu13 = Path(torch.__file__).resolve().parent.parent / "nvidia" / "cu13"
    if not (cu13 / "bin" / "nvcc").exists():
        return
    os.environ["CUDA_HOME"] = str(cu13)
    os.environ["PATH"] = str(cu13 / "bin") + os.pathsep + os.environ.get("PATH", "")
    if "torch.utils.cpp_extension" in sys.modules:
        sys.modules["torch.utils.cpp_extension"].CUDA_HOME = str(cu13)


def _load():
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load

    _ensure_nvcc()
    src_dir = Path(__file__).resolve().parent
    build_dir = src_dir / "build"
    build_dir.mkdir(exist_ok=True)
    _ext = load(
        name="blackwell_ops_cuda",
        sources=[
            str(src_dir / "basic_ops.cu"),
            str(src_dir / "kv_quant.cu"),
            str(src_dir / "lora_ops.cu"),
            str(src_dir / "lora_tiled.cu"),
            str(src_dir / "mla_decode.cu"),
            str(src_dir / "cutlass_gemm.cu"),
        ],
        extra_include_paths=[
            "/home/qiaosir/projects_1/kda_cuda_rtx5070ti/cutlass/include"
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_120,code=sm_120",
        ],
        build_directory=str(build_dir),
        verbose=False,
    )
    return _ext


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _load().rmsnorm(x, weight, eps)


def fused_residual_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return _load().fused_residual_rmsnorm(x, residual, weight, eps)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return _load().rope(x, cos, sin)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return _load().swiglu(gate, up)


def quantize_kv_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x: [C, N] fp16 -> (q: [C, N] int8, scale: [C] fp32)"""
    q, scale = _load().quantize_kv_int8(x)
    return q, scale


def dequantize_kv_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """q: [C, N] int8, scale: [C] fp32 -> x: [C, N] fp16"""
    return _load().dequantize_kv_int8(q, scale)


def lora_delta(A: torch.Tensor, B: torch.Tensor, scaling: float = 1.0) -> torch.Tensor:
    """A: [r, K], B: [N, r] -> delta: [N, K] = B @ A * scaling"""
    return _load().lora_delta(A, B, scaling)


def add_delta_(W: torch.Tensor, delta: torch.Tensor) -> None:
    """W += delta, in-place."""
    _load().add_delta_(W, delta)


def lora_delta_tiled(
    A: torch.Tensor, B: torch.Tensor, scaling: float = 1.0
) -> torch.Tensor:
    """A: [r, K], B: [N, r] -> delta: [N, K] = B @ A * scaling (tiled CUDA)"""
    return _load().lora_delta_tiled(A, B, scaling)


def mla_decode(Q: torch.Tensor, cKV: torch.Tensor) -> torch.Tensor:
    """Q: [B, H, L] (weight-absorbed query), cKV: [B, N, L] -> O: [B, H, L]"""
    return _load().mla_decode(Q, cKV)


def cutlass_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A: [M, K], B: [N, K] -> C: [M, N] = A @ B.T (CUTLASS FP16)"""
    return _load().cutlass_gemm(A, B)
