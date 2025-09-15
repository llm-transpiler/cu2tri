import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for HardSwish"""
    return torch.nn.functional.hardswish(x)
