"""Minimal attention backend demo."""

import torch

from blackwell_inference.attention import forward

B, H, N, D = 1, 4, 512, 64
q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

for backend in ["sdpa", "linear", "kda"]:
    out = forward(backend, q, k, v, causal=True)
    print(f"{backend}: out shape={out.shape} max={out.abs().max().item():.4f}")
