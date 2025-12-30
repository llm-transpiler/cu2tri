#!/usr/bin/env python3
"""
Jensen-Shannon Divergence Benchmark - Liger-Kernel Implementation
基于Liger-Kernel的JS散度实现
"""

import torch
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Params:
    """Jensen-Shannon Divergence 测试参数"""
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 覆盖不同模型规模的测试用例
            self.test_shapes = [
                # Small models
                (16, 512),          # Small batch, small vocab
                (32, 1000),         # Medium batch, medium vocab
                (64, 2000),         # Large batch, large vocab

                # Medium models
                (8, 5000),          # Small batch, large vocab
                (16, 10000),        # Medium batch, very large vocab
                (32, 32000),        # Large batch, Llama vocab size

                # Large models
                (4, 50000),         # Small batch, huge vocab
                (8, 100000),        # Medium batch, massive vocab

                # 3D tensors
                (2, 16, 1000),      # Small batch, short seq
                (4, 32, 5000),      # Medium batch, medium seq
                (1, 64, 10000),     # Single batch, long seq

                # Edge cases
                (1, 10),            # Minimal
                (128, 100),         # Large batch, small vocab
            ]

def get_cuda_argtypes():
    """CUDA类型定义"""
    return ["float32", "float32"]

def get_cuda_torch_inputs(params: Params):
    """生成单个测试用例的输入数据"""
    shape = params.test_shapes[0]

    # Generate two probability distributions
    p = torch.randn(shape, dtype=torch.float32, device='cuda')
    q = torch.randn(shape, dtype=torch.float32, device='cuda')

    # Normalize to probability distributions
    p = torch.softmax(p, dim=-1)
    q = torch.softmax(q, dim=-1)

    return [p, q]

def get_all_cuda_torch_inputs(params: Params) -> List[Tuple[Tuple[int, ...], Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]]:
    """生成所有测试形状的输入数据"""
    test_cases = []

    for i, shape in enumerate(params.test_shapes):
        # Generate two probability distributions
        p = torch.randn(shape, dtype=torch.float32, device='cuda')
        q = torch.randn(shape, dtype=torch.float32, device='cuda')

        # Normalize to probability distributions
        p = torch.softmax(p, dim=-1)
        q = torch.softmax(q, dim=-1)

        cuda_inputs = [p, q]
        torch_inputs = [p.clone(), q.clone()]

        cuda_output_tensors = []

        test_cases.append((shape, (cuda_inputs, torch_inputs, cuda_output_tensors)))

    return test_cases

def torch_kernel(p: torch.Tensor, q: torch.Tensor, reduction: str = "batchmean") -> torch.Tensor:
    """
    PyTorch参考实现：Jensen-Shannon Divergence
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)，其中M = 0.5 * (P + Q)

    Args:
        p: First probability distribution
        q: Second probability distribution
        reduction: Reduction method

    Returns:
        Jensen-Shannon divergence
    """
    # M = 0.5 * (P + Q)
    m = 0.5 * (p + q)

    # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    kl_pm = torch.sum(p * (torch.log(p + 1e-8) - torch.log(m + 1e-8)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q + 1e-8) - torch.log(m + 1e-8)), dim=-1)

    jsd = 0.5 * (kl_pm + kl_qm)

    if reduction == "none":
        return jsd
    elif reduction == "sum":
        return jsd.sum()
    elif reduction == "mean":
        return jsd.mean()
    elif reduction == "batchmean":
        batch_size = p.shape[0]
        return jsd.view(batch_size, -1).mean(dim=1).mean()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def test_cases():
    """提供测试用例兼容性"""
    params = Params()
    return get_all_cuda_torch_inputs(params)

def torch_jsd(p: torch.Tensor, q: torch.Tensor, reduction: str = "batchmean") -> torch.Tensor:
    """兼容性别名"""
    return torch_kernel(p, q, reduction)