# import torch

# def torch_kernel(x: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     """PyTorch reference implementation for LayerNorm"""
#     return torch.nn.functional.layer_norm(x, x.shape[-1:], g, b)
import torch
import torch.nn.functional as F

def torch_kernel(x: torch.Tensor, g: float, b: float) -> torch.Tensor:
    """PyTorch reference implementation for layer norm f16x8_pack"""
    # Apply layer normalization
    # PyTorch layer norm: (x - mean) / sqrt(var + eps) * gamma + beta
    x_normalized = F.layer_norm(x.float(), x.shape[-1:], weight=None, bias=None, eps=1e-5)
    
    # Apply scale and bias
    result = x_normalized * g + b
    
    # Return in same precision as input
    return result.half()
