import torch

def torch_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for dot product"""
    return torch.dot(a, b)
