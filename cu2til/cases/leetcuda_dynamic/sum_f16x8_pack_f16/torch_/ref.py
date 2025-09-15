import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for sum"""
    return torch.sum(x)
