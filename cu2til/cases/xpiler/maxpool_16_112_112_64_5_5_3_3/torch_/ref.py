import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.max_pool2d(x, kernel_size=5, stride=3)
