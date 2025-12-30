import torch

def torch_kernel(x: torch.Tensor, p: float) -> torch.Tensor:
    """PyTorch参考实现：dropout"""
    # PyTorch的dropout实现
    return torch.nn.functional.dropout(x, p=p, training=True)