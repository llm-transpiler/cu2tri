import torch
import torch.nn.functional as F

def torch_kernel(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    PyTorch参考实现：Softmax
    计算softmax激活函数

    Args:
        input_tensor: Input tensor
        dim: Dimension along which to compute softmax

    Returns:
        Softmax of the input tensor
    """
    return F.softmax(input_tensor, dim=dim)