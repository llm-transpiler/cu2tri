#!/usr/bin/env python3
"""
Softmax Benchmark - Liger-Kernel Implementation
基于Liger-Kernel的Softmax实现
"""

import torch
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Params:
    """Softmax 测试参数"""
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 覆盖不同模型规模的测试用例
            self.test_shapes = [
                # Small dimensions
                (16, 64),           # Small batch, small features
                (32, 128),          # Medium batch, small features
                (64, 256),          # Medium batch, medium features

                # Medium dimensions (common in attention)
                (8, 1024),          # Small batch, medium features
                (16, 2048),         # Small batch, large features
                (4, 4096),          # Tiny batch, very large features

                # Large dimensions
                (2, 8192),          # Tiny batch, huge features
                (1, 16384),         # Single token, massive features
                (32, 512),          # Large batch, medium features

                # Attention-like dimensions
                (64, 128),          # Sequence attention
                (128, 256),         # Medium sequence attention
                (256, 512),         # Large sequence attention

                # 3D tensors (batch, seq_len, vocab_size)
                (4, 16, 1000),      # Small batch, short seq, small vocab
                (2, 32, 32000),     # Small batch, medium seq, large vocab
                (1, 64, 50000),     # Single batch, long seq, huge vocab

                # Edge cases
                (1, 1),             # Single element
                (1, 10),            # Very small features
                (100, 10),          # Large batch, tiny features
            ]

def get_cuda_argtypes():
    """CUDA类型定义"""
    return ["float32"]

def get_cuda_torch_inputs(params: Params):
    """生成单个测试用例的输入数据"""
    shape = params.test_shapes[0]

    # Generate input with reasonable range to avoid overflow/underflow
    input_tensor = torch.randn(shape, dtype=torch.float32, device='cuda') * 2.0

    return [input_tensor]

def get_all_cuda_torch_inputs(params: Params) -> List[Tuple[Tuple[int, ...], Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]]:
    """生成所有测试形状的输入数据"""
    test_cases = []

    for i, shape in enumerate(params.test_shapes):
        # Generate test data with controlled range
        input_tensor = torch.randn(shape, dtype=torch.float32, device='cuda') * 1.0

        # Clamp to prevent extreme values that cause numerical issues
        input_tensor = torch.clamp(input_tensor, -10.0, 10.0)

        cuda_inputs = [input_tensor]
        torch_inputs = [input_tensor.clone()]

        # Create output tensors
        cuda_output_tensors = []

        test_cases.append((shape, (cuda_inputs, torch_inputs, cuda_output_tensors)))

    return test_cases

def torch_kernel(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    PyTorch参考实现：Softmax
    """
    return torch.softmax(input_tensor, dim=dim)

def test_cases():
    """提供测试用例兼容性"""
    params = Params()
    return get_all_cuda_torch_inputs(params)

def torch_softmax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """兼容性别名"""
    return torch_kernel(input_tensor, dim)