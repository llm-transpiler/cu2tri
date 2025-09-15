import torch

def torch_kernel(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for RMSNorm"""
    # RMSNorm: x / sqrt(mean(x^2) + eps) * g
    eps = 1e-5
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * g
