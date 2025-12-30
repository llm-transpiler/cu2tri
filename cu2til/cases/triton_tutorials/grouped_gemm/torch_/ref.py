import torch
from typing import List

def torch_kernel(group_A: List[torch.Tensor], group_B: List[torch.Tensor]) -> List[torch.Tensor]:
    """PyTorch参考实现：grouped matrix multiplication"""
    group_C = []
    for A, B in zip(group_A, group_B):
        C = torch.matmul(A, B)
        group_C.append(C)
    return group_C