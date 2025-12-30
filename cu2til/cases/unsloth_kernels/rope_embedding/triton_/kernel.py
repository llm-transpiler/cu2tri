import torch
import triton
import triton.language as tl

def triton_kernel(Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, output: torch.Tensor,
                  batch_size: int, seq_len: int, n_heads: int, head_dim: int) -> torch.Tensor:
    """
    Triton实现：RoPE embedding
    RoPE is Q * cos + rotate_half(Q) * sin
    """
    # 分割Q用于rotate_half
    Q1, Q2 = Q[..., :head_dim//2], Q[..., head_dim//2:]
    Q_rotated = torch.cat([-Q2, Q1], dim=-1)

    # 扩展cos和sin以匹配Q的维度
    # cos和sin的形状为[seq_len, head_dim//2]，需要扩展到[batch_size, seq_len, n_heads, head_dim//2]
    cos_expanded = cos.unsqueeze(0).unsqueeze(2).expand(batch_size, seq_len, n_heads, -1)
    sin_expanded = sin.unsqueeze(0).unsqueeze(2).expand(batch_size, seq_len, n_heads, -1)

    # 拼接cos和sin以获得完整的head维度
    cos_full = torch.cat([cos_expanded, cos_expanded], dim=-1)
    sin_full = torch.cat([sin_expanded, sin_expanded], dim=-1)

    # 应用RoPE公式
    output.copy_(Q * cos_full + Q_rotated * sin_full)

    return output

def triton_rope_embedding(Q, cos, sin):
    """保持向后兼容的函数名"""
    head_dim = Q.shape[-1]
    batch_size, seq_len, n_heads, _ = Q.shape

    # 分割Q用于rotate_half
    Q1, Q2 = Q[..., :head_dim//2], Q[..., head_dim//2:]
    Q_rotated = torch.cat([-Q2, Q1], dim=-1)

    # 扩展cos和sin以匹配Q的维度
    cos_expanded = cos.unsqueeze(0).unsqueeze(2).expand(batch_size, seq_len, n_heads, -1)
    sin_expanded = sin.unsqueeze(0).unsqueeze(2).expand(batch_size, seq_len, n_heads, -1)

    # 拼接cos和sin以获得完整的head维度
    cos_full = torch.cat([cos_expanded, cos_expanded], dim=-1)
    sin_full = torch.cat([sin_expanded, sin_expanded], dim=-1)

    # 应用RoPE公式
    Q_rope = Q * cos_full + Q_rotated * sin_full

    return Q_rope