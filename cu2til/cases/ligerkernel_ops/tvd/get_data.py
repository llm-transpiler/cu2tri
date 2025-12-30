#!/usr/bin/env python3
"""
Total Variation Distance Benchmark - Liger-Kernel Implementation
基于Liger-Kernel的全变差距离实现
"""

import torch
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Params:
    """TVD 测试参数"""
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            self.test_shapes = [
                (16, 512), (32, 1000), (64, 2000),
                (8, 5000), (16, 10000), (32, 32000),
                (4, 50000), (8, 100000),
                (2, 16, 1000), (4, 32, 5000), (1, 64, 10000),
                (1, 10), (128, 100), (256, 256),
            ]

def get_cuda_argtypes():
    return ["float32", "float32"]

def get_cuda_torch_inputs(params: Params):
    shape = params.test_shapes[0]
    p = torch.randn(shape, dtype=torch.float32, device='cuda')
    q = torch.randn(shape, dtype=torch.float32, device='cuda')
    p = torch.softmax(p, dim=-1)
    q = torch.softmax(q, dim=-1)
    return [p, q]

def get_all_cuda_torch_inputs(params: Params):
    test_cases = []
    for shape in params.test_shapes:
        p = torch.randn(shape, dtype=torch.float32, device='cuda')
        q = torch.randn(shape, dtype=torch.float32, device='cuda')
        p = torch.softmax(p, dim=-1)
        q = torch.softmax(q, dim=-1)
        test_cases.append((shape, ([p, q], [p.clone(), q.clone()], [])))
    return test_cases

def torch_kernel(p: torch.Tensor, q: torch.Tensor, reduction: str = "batchmean") -> torch.Tensor:
    """
    PyTorch参考实现：Total Variation Distance
    TVD(P||Q) = 0.5 * sum(|P - Q|)
    """
    tvd = 0.5 * torch.sum(torch.abs(p - q), dim=-1)

    if reduction == "none":
        return tvd
    elif reduction == "sum":
        return tvd.sum()
    elif reduction == "mean":
        return tvd.mean()
    elif reduction == "batchmean":
        batch_size = p.shape[0]
        return tvd.view(batch_size, -1).mean(dim=1).mean()
    return tvd

def test_cases():
    params = Params()
    return get_all_cuda_torch_inputs(params)

def torch_tvd(p: torch.Tensor, q: torch.Tensor, reduction: str = "batchmean") -> torch.Tensor:
    return torch_kernel(p, q, reduction)