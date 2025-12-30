#!/usr/bin/env python3
"""
Layer Normalization Benchmark - Liger-Kernel Implementation
基于Liger-Kernel的高性能LayerNorm实现
"""

import torch
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Params:
    """Layer Normalization 测试参数"""
    test_shapes: List[Tuple[int, int]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 覆盖不同模型规模的测试用例 (batch_size, hidden_size)
            self.test_shapes = [
                # Small models
                (2, 512),         # Small batch, small hidden
                (4, 768),         # Medium batch, medium hidden
                (8, 1024),        # Larger batch, large hidden

                # Medium models (Llama-7B style)
                (2, 1024),        # Small batch, medium hidden
                (4, 2048),        # Medium batch, large hidden
                (8, 4096),        # Larger batch, very large hidden

                # Large models (Llama-70B style)
                (1, 4096),        # Small batch, large hidden
                (2, 8192),        # Small batch, very large hidden
                (4, 16384),       # Medium batch, huge hidden

                # Special cases
                (16, 256),        # Large batch, tiny hidden
                (32, 128),        # Very large batch, tiny hidden
                (1, 32768),       # Single sample, massive hidden

                # Sequence length variations (treat seq_len as batch_size)
                (128, 512),       # Many sequences, small hidden
                (64, 1024),       # Many sequences, medium hidden
                (32, 2048),       # Many sequences, large hidden
            ]

def get_cuda_argtypes():
    """CUDA类型定义"""
    return ["float32", "float32", "float32", "float32"]

def get_cuda_torch_inputs(params: Params):
    """生成单个测试用例的输入数据"""
    batch_size, hidden_size = params.test_shapes[0]

    # Generate random input
    X = torch.randn(batch_size, hidden_size, dtype=torch.float32, device='cuda')

    # Generate weight and bias
    W = torch.randn(hidden_size, dtype=torch.float32, device='cuda')
    B = torch.randn(hidden_size, dtype=torch.float32, device='cuda')

    return [X, W, B]

def get_all_cuda_torch_inputs(params: Params) -> List[Tuple[Tuple[int, ...], Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]]:
    """生成所有测试形状的输入数据"""
    test_cases = []

    for i, shape in enumerate(params.test_shapes):
        batch_size, hidden_size = shape

        # Generate test data with reasonable ranges
        X = torch.randn(batch_size, hidden_size, dtype=torch.float32, device='cuda') * 0.5
        W = torch.randn(hidden_size, dtype=torch.float32, device='cuda') * 0.1
        B = torch.randn(hidden_size, dtype=torch.float32, device='cuda') * 0.1

        cuda_inputs = [X, W, B]
        torch_inputs = [X.clone(), W.clone(), B.clone()]

        # Create output tensors
        cuda_output_tensors = []

        test_cases.append((shape, (cuda_inputs, torch_inputs, cuda_output_tensors)))

    return test_cases

def torch_kernel(X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """PyTorch参考实现：Layer Normalization"""
    return torch.nn.functional.layer_norm(X, X.shape[-1:], weight=W, bias=B, eps=eps)