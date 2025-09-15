import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for Swish"""
    return x * torch.sigmoid(x)
