import torch

def torch_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for HGEMM"""
    return torch.matmul(A, B)
