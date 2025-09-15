import torch

def torch_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for elementwise add"""
    return a + b
