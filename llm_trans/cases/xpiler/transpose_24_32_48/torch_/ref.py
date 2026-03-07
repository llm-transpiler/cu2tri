import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return x.permute(2, 0, 1).contiguous()
