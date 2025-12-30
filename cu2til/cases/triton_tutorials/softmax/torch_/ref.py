import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现：fused softmax"""
    return torch.softmax(x, dim=-1)