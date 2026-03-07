import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def torch_kernel(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Grouped Query Attention (GQA) using F.scaled_dot_product_attention.
    Q: [batch*num_q_heads, M, D]   e.g., [2, 16, 16]
    K: [batch*num_kv_heads, N, D]  e.g., [1, 512, 16]
    V: [batch*num_kv_heads, N, D]  e.g., [1, 512, 16]
    
    Uses repeat_interleave to expand K/V to match Q heads for GQA.
    """
    num_q_heads = Q.shape[0]
    num_kv_heads = K.shape[0]
    group_size = num_q_heads // num_kv_heads
    
    # Convert to float32 for numerical stability
    Q = Q.to(torch.float32)
    K = K.to(torch.float32)
    V = V.to(torch.float32)
    
    # Expand K and V to match Q heads via repeat_interleave
    # Each KV head is repeated group_size times
    # K: [num_kv_heads, N, D] -> [num_q_heads, N, D]
    K_expanded = K.repeat_interleave(group_size, dim=0)
    V_expanded = V.repeat_interleave(group_size, dim=0)
    
    # Add head dimension: [num_q_heads, M/N, D] -> [1, num_q_heads, M/N, D]
    Q = Q.unsqueeze(0)  # [1, num_q_heads, M, D]
    K_expanded = K_expanded.unsqueeze(0)  # [1, num_q_heads, N, D]
    V_expanded = V_expanded.unsqueeze(0)  # [1, num_q_heads, N, D]
    
    # Use SDPA with standard scaling (1/sqrt(d))
    # Force MATH backend for determinism across GPUs
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(
            Q, K_expanded, V_expanded,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None  # Use default 1/sqrt(d) scaling
        )
    
    # Remove batch dimension and convert back to original dtype
    out = out.squeeze(0)  # [num_q_heads, M, D]
    return out.to(torch.float16)



