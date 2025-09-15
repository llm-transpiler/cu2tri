import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for ELU"""
    return torch.nn.functional.elu(x)
