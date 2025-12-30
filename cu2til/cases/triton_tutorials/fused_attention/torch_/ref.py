import torch
from typing import Tuple

def torch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """PyTorch参考实现：scaled dot-product attention"""
    # 使用torch.nn.functional.scaled_dot_product_attention (PyTorch 2.0+)
    try:
        import torch.nn.functional as F
        # Reshape for scaled_dot_product_attention: (B, H, S, D) -> (B, S, H, D)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # 使用PyTorch的优化attention实现
        attn_output = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            dropout_p=0.0,
            is_causal=False,
            scale=scale
        )
        # Reshape back: (B, S, H, D) -> (B, H, S, D)
        return attn_output.transpose(1, 2)
    except ImportError:
        # Manual implementation for older PyTorch versions
        batch_size, n_heads, seq_len, head_dim = q.shape

        # Reshape for matrix multiplication: (B, H, S, D) -> (B*H, S, D)
        q_flat = q.view(batch_size * n_heads, seq_len, head_dim)
        k_flat = k.view(batch_size * n_heads, seq_len, head_dim)
        v_flat = v.view(batch_size * n_heads, seq_len, head_dim)

        # Compute attention scores: (B*H, S, S)
        scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * scale

        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values: (B*H, S, D)
        attn_output = torch.matmul(attn_weights, v_flat)

        # Reshape back: (B*H, S, D) -> (B, H, S, D)
        return attn_output.view(batch_size, n_heads, seq_len, head_dim)