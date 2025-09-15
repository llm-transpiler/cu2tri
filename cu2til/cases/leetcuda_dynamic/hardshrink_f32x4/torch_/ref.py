import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for HardShrink"""
    return torch.nn.functional.hardshrink(x)
