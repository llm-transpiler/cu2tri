import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for ReLU"""
    return torch.nn.functional.relu(x)
