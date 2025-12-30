import torch
import torch.nn.functional as F

def torch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention via PyTorch scaled_dot_product_attention.

    Expects q, k, v shaped as (B, N_CTX, H, D_HEAD) and returns the same shape.
    """
    B, N_CTX, H, D_HEAD = q.shape

    # Use SDPA across the head dimension H (head-mixing per token):
    # Treat original H as sequence length (S=L=H) and set number of heads to 1.
    # Shapes for SDPA: (B, N_CTX, heads=1, L=H, E)
    q_t = q.unsqueeze(2)
    k_t = k.unsqueeze(2)
    v_t = v.unsqueeze(2)

    out = F.scaled_dot_product_attention(q_t, k_t, v_t)

    # Squeeze back heads dim -> (B, N_CTX, H, D_HEAD)
    return out.squeeze(2)
