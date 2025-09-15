import torch

def torch_kernel(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for HGEMV"""
    return torch.mv(a, x)
