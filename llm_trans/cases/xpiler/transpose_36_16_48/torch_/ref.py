import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1).contiguous()
