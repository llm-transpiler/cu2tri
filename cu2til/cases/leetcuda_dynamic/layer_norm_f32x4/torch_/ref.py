# import torch

# def torch_kernel(x: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     """PyTorch reference implementation for LayerNorm"""
#     return torch.nn.functional.layer_norm(x, x.shape[-1:], g, b)
import torch
import torch.nn.functional as F

def torch_kernel(x: torch.Tensor, g: float, b: float) -> torch.Tensor:
    """PyTorch reference implementation for Layer Normalization"""
    # Apply layer norm along the last dimension
    # normalized_shape should be the size of the last dimension
    layer_norm = torch.nn.LayerNorm(x.shape[-1], eps=1e-5, elementwise_affine=False, device=x.device)
    normalized = layer_norm(x)
    # Apply scale and bias
    return normalized * g + b
