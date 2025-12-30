import torch

def torch_kernel(x: torch.Tensor, alpha: float = 1.0, beta: float = 1.0) -> torch.Tensor:
    """Polynomial Normalization: y = (x + beta)^alpha"""
    return torch.clamp(x + beta, min=1e-8) ** alpha