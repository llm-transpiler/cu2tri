import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for Online Safe Softmax"""
    return torch.nn.functional.softmax(x, dim=-1)
