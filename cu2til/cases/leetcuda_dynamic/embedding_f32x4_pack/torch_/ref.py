import torch

def torch_kernel(idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for embedding"""
    return torch.nn.functional.embedding(idx, weight)
