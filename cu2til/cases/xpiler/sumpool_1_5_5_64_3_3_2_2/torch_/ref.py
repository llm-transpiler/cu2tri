import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=2) * 9
