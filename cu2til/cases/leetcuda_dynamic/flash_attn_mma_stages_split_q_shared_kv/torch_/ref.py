import torch

def torch_kernel(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for Flash Attention"""
    # Q, K, V: [batch_size, num_heads, seq_len, head_dim]
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
    attn = torch.nn.functional.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out
