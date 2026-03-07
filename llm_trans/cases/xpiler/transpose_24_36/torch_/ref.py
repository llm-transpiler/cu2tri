import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return x.permute(1, 0).contiguous()
