import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def torch_kernel(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    # Q: [bh, M, D], K: [bh, N, D], V: [bh, N, D]
    # Align to PyTorch SDPA semantics WITHOUT scaling (scale=1.0)
    q = Q.to(torch.float32).unsqueeze(1)  # [bh, 1, M, D]
    k = K.to(torch.float32).unsqueeze(1)  # [bh, 1, N, D]
    v = V.to(torch.float32).unsqueeze(1)  # [bh, 1, N, D]
    # Force math backend for determinism across GPUs; disable scale (scale=1.0)
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)
    return out.squeeze(1).to(Q.dtype)

