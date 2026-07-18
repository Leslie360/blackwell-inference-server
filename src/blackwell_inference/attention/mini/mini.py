"""Optional SM120 flash-attention backend adapted from ShlokVFX/Mini-Attention.

Set ``MINI_ATTENTION_ROOT`` to the cloned Mini-Attention repository before use.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

_DEFAULT_ROOT = Path(os.environ.get("MINI_ATTENTION_ROOT", "/home/qiaosir/projects_1/kda_cuda_rtx5070ti/mini_attention"))
_ext = None
_cfg = None


def _load():
    global _ext, _cfg
    if _ext is not None:
        return _ext, _cfg

    so_path = _DEFAULT_ROOT / "kernels" / "flash_attn" / "cuda" / "sm_120" / "build" / "sm120" / "sm120.so"
    py_dir = _DEFAULT_ROOT / "kernels" / "flash_attn" / "cuda" / "sm_120" / "py"
    if not so_path.exists():
        raise RuntimeError(
            f"Mini-Attention extension not found at {so_path}. "
            "Build it first or set MINI_ATTENTION_ROOT."
        )

    if str(py_dir) not in sys.path:
        sys.path.insert(0, str(py_dir))
    from flash_helpers.kernel_configs import DType, FlashForwardKernelConfig

    spec = importlib.util.spec_from_file_location("sm120", str(so_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    _ext = mod
    _cfg = FlashForwardKernelConfig(DType.FP16, 128, 64, 64, 4, True, True, True, 0, 2, 2, True, True)
    return _ext, _cfg


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Non-causal flash attention.

    q, k, v: [B, H, N, D] fp16, D=128
    returns O: [B, H, N, D]
    """
    ext, cfg = _load()
    assert q.dtype == torch.float16 and q.shape[-1] == 128
    q_bnhd = q.transpose(1, 2).contiguous()
    k_bnhd = k.transpose(1, 2).contiguous()
    v_bnhd = v.transpose(1, 2).contiguous()
    o_bnhd = torch.empty_like(q_bnhd)
    ext.forward(cfg, q_bnhd, k_bnhd, v_bnhd, o_bnhd, False)
    return o_bnhd.transpose(1, 2).contiguous()
