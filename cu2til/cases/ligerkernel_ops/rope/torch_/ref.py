import torch
import math

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0, device='cuda') -> torch.Tensor:
    """
    Precompute frequency tensor for complex exponentials (cos, sin)

    Args:
        dim: Dimension of the embedding
        seq_len: Sequence length
        theta: Base frequency
        device: Device to compute on

    Returns:
        Tuple of (cos, sin) tensors of shape (seq_len, dim//2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors

    Args:
        xq: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
        xk: Key tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)
        freqs_cos: Cosine frequencies of shape (seq_len, head_dim//2)
        freqs_sin: Sine frequencies of shape (seq_len, head_dim//2)

    Returns:
        Tuple of rotated (query, key) tensors
    """
    # Reshape for broadcasting - handle different seq_len in freqs
    seq_len_q = xq.shape[1]
    seq_len_k = xk.shape[1]

    # Use appropriate slice of freqs based on actual seq_len
    freqs_cos_q = freqs_cos[:seq_len_q, :].unsqueeze(1).unsqueeze(0)  # (1, seq_len_q, 1, head_dim//2)
    freqs_sin_q = freqs_sin[:seq_len_q, :].unsqueeze(1).unsqueeze(0)
    freqs_cos_k = freqs_cos[:seq_len_k, :].unsqueeze(1).unsqueeze(0)
    freqs_sin_k = freqs_sin[:seq_len_k, :].unsqueeze(1).unsqueeze(0)

    head_dim = xq.shape[-1]

    # Apply rotation formula:
    # [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    xq_cos = xq * freqs_cos_q
    xq_sin = xq * freqs_sin_q

    # Split into two halves along head dimension
    xq1, xq2 = xq_cos[..., :head_dim//2], xq_cos[..., head_dim//2:]
    xq3, xq4 = xq_sin[..., :head_dim//2], xq_sin[..., head_dim//2:]

    # Apply rotation: real part and imaginary part
    xq_new = torch.cat([xq1 - xq4, xq2 + xq3], dim=-1)

    # Same for keys
    xk_cos = xk * freqs_cos_k
    xk_sin = xk * freqs_sin_k

    xk1, xk2 = xk_cos[..., :head_dim//2], xk_cos[..., head_dim//2:]
    xk3, xk4 = xk_sin[..., :head_dim//2], xk_sin[..., head_dim//2:]

    xk_new = torch.cat([xk1 - xk4, xk2 + xk3], dim=-1)

    return xq_new, xk_new

def torch_kernel(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch参考实现：RoPE (Rotary Position Embedding)
    对查询和键张量应用旋转位置编码

    Args:
        q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)

    Returns:
        Tuple of rotated (query, key) tensors
    """
    batch_size, seq_len_q, num_heads, head_dim = q.shape
    _, seq_len_k, num_kv_heads, _ = k.shape

    # Use the maximum sequence length for precomputing frequencies
    max_seq_len = max(seq_len_q, seq_len_k)

    # Precompute cos and sin values
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, max_seq_len, device=q.device)

    # Apply rotary embeddings
    q_new, k_new = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

    return q_new, k_new