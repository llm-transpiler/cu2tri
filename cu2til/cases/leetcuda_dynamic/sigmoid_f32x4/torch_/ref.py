import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for Sigmoid"""
    return torch.sigmoid(x)
