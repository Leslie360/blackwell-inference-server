"""Lazy-built raw CUDA causal attention kernel (KDA)."""

from __future__ import annotations

from pathlib import Path

import torch

_KERNEL = None


def _load_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL

    try:
        from torch.utils.cpp_extension import load
    except ImportError as e:
        raise RuntimeError("torch.utils.cpp_extension is required for KDA kernel") from e

    src = Path(__file__).resolve().parent / "kda_attention.cu"
    build_dir = src.parent / "build"
    build_dir.mkdir(exist_ok=True)

    _KERNEL = load(
        name="blackwell_kda",
        sources=[str(src)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_120,code=sm_120",
        ],
        build_directory=str(build_dir),
        verbose=False,
    )
    return _KERNEL


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Causal softmax attention.

    q, k, v: [B, H, N, D] fp16/bf16, D=64
    returns O: [B, H, N, D]
    """
    kernel = _load_kernel()
    return kernel.forward(q, k, v)
