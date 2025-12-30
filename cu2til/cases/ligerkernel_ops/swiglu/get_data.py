#!/usr/bin/env python3
"""
SwiGLU Benchmark - Liger-Kernel Implementation
基于Liger-Kernel的Swish-Gated Linear Unit实现
"""

import torch
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Params:
    """SwiGLU 测试参数"""
    test_shapes: List[Tuple[int, int, int]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 覆盖不同模型规模的测试用例 (batch, seq_len, hidden_dim)
            self.test_shapes = [
                # Small models
                (2, 128, 512),     # Small batch, short sequence, small hidden
                (4, 256, 768),     # Medium batch, medium sequence, medium hidden
                (8, 512, 1024),    # Larger batch, longer sequence, large hidden

                # Medium models (Llama-7B style)
                (2, 256, 1024),    # Small batch, medium sequence, medium hidden
                (4, 512, 2048),    # Medium batch, long sequence, large hidden
                (8, 1024, 4096),   # Larger batch, very long sequence, very large hidden

                # Large models (Llama-70B style)
                (1, 128, 4096),    # Small batch, short sequence, large hidden
                (2, 256, 8192),    # Small batch, medium sequence, very large hidden
                (4, 512, 16384),   # Medium batch, long sequence, huge hidden

                # Very large models
                (1, 64, 16384),    # Tiny sequence, huge hidden
                (2, 128, 32768),   # Small batch, short sequence, massive hidden

                # Special cases
                (16, 32, 256),     # Large batch, tiny sequence, small hidden
                (32, 16, 512),     # Very large batch, tiny sequence, medium hidden

                # Sequence length variations
                (1, 4096, 1024),   # Very long sequence
                (1, 8192, 512),    # Extremely long sequence
                (2, 2048, 2048),   # Long sequence, large hidden
            ]

def get_cuda_argtypes():
    """CUDA类型定义"""
    return ["float32", "float32"]

def get_cuda_torch_inputs(params: Params):
    """生成单个测试用例的输入数据"""
    batch_size, seq_len, hidden_dim = params.test_shapes[0]

    # Generate random gate and up projections
    gate = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device='cuda')
    up = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device='cuda')

    return [gate, up]

def get_all_cuda_torch_inputs(params: Params) -> List[Tuple[Tuple[int, ...], Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]]:
    """生成所有测试形状的输入数据"""
    test_cases = []

    for i, shape in enumerate(params.test_shapes):
        batch_size, seq_len, hidden_dim = shape

        # Generate test data with reasonable ranges
        gate = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device='cuda') * 0.5
        up = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device='cuda') * 0.5

        cuda_inputs = [gate, up]
        torch_inputs = [gate.clone(), up.clone()]

        # Create output tensors
        cuda_output_tensors = []

        test_cases.append((shape, (cuda_inputs, torch_inputs, cuda_output_tensors)))

    return test_cases

def torch_kernel(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    PyTorch参考实现：SwiGLU (Swish-Gated Linear Unit)
    SwiGLU(gate, up) = Swish(gate) * up where Swish(x) = x * sigmoid(x)
    """
    # Swish activation: f = gate * sigmoid(gate)
    swish = gate * torch.sigmoid(gate)
    # SwiGLU: h = swish * up
    return swish * up