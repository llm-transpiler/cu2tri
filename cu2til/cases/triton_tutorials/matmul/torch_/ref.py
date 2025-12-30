import torch

def torch_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现：matrix multiplication"""
    return torch.matmul(a, b)