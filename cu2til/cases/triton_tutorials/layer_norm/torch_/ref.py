import torch
from typing import Tuple

def torch_kernel(x: torch.Tensor, normalized_shape: Tuple[int, ...], weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """PyTorch参考实现：layer normalization"""
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)