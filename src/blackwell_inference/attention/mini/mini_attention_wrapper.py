"""Wrapper for ShlokVFX/Mini-Attention SM120 flash attention kernel.

Loads the prebuilt extension and adapts [B,H,N,D] layout to the kernel's
[B,N,H,D] layout. Used as the high-performance reference/optimized KDA.
"""

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
MINI_ROOT = ROOT / "mini_attention"
MINI_SO = (
    MINI_ROOT
    / "kernels"
    / "flash_attn"
    / "cuda"
    / "sm_120"
    / "build"
    / "sm120"
    / "sm120.so"
)
MINI_PY = MINI_ROOT / "kernels" / "flash_attn" / "cuda" / "sm_120" / "py"

if str(MINI_PY) not in sys.path:
    sys.path.insert(0, str(MINI_PY))

from flash_helpers.kernel_configs import DType, FlashForwardKernelConfig  # noqa: E402


def _load_ext():
    spec = importlib.util.spec_from_file_location("sm120", str(MINI_SO))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension from {MINI_SO}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ext = _load_ext()

# Best config from Mini-Attention benchmark on SM120: K7 auto-tuned
_K7_CFG = FlashForwardKernelConfig(
    DType.FP16, 128, 64, 64, 4, True, True, True, 0, 2, 2, True, True
)


def mini_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    q,k,v: [B, H, N, D] fp16, D=128
    returns O: [B, H, N, D]
    """
    assert q.dtype == torch.float16 and q.shape[-1] == 128
    q_bnhd = q.transpose(1, 2).contiguous()
    k_bnhd = k.transpose(1, 2).contiguous()
    v_bnhd = v.transpose(1, 2).contiguous()
    o_bnhd = torch.empty_like(q_bnhd)
    _ext.forward(_K7_CFG, q_bnhd, k_bnhd, v_bnhd, o_bnhd, False)
    return o_bnhd.transpose(1, 2).contiguous()
