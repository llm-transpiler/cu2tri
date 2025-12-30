#!/usr/bin/env python3
"""
Fused Add RMS Norm Benchmark - Liger-Kernel Implementation
基于Liger-Kernel的融合加法RMS归一化实现
"""

import torch
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Params:
    """Fused Add RMS Norm 测试参数"""
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 覆盖不同模型规模的测试用例
            self.test_shapes = [
                # Small models
                (16, 128),          # Small batch, small hidden
                (32, 256),          # Medium batch, medium hidden
                (64, 512),          # Large batch, large hidden

                # Medium models (Llama-7B style)
                (8, 1024),          # Small batch, medium hidden
                (16, 2048),         # Medium batch, large hidden
                (32, 4096),         # Large batch, very large hidden

                # Large models (Llama-70B style)
                (4, 4096),          # Small batch, large hidden
                (8, 8192),          # Medium batch, very large hidden
                (16, 16384),        # Large batch, huge hidden

                # 3D tensors (batch, seq_len, hidden_dim)
                (2, 16, 512),       # Small batch, short seq, small hidden
                (4, 32, 1024),      # Small batch, medium seq, medium hidden
                (1, 64, 2048),      # Single batch, long seq, large hidden

                # Special cases
                (128, 64),          # Very large batch, tiny hidden
                (1, 32768),         # Single token, massive hidden
                (256, 32),          # Huge batch, very small hidden

                # Sequence length variations
                (1, 128, 1024),     # Single batch, short sequence
                (1, 512, 1024),     # Single batch, medium sequence
                (1, 2048, 1024),    # Single batch, long sequence
            ]

def get_cuda_argtypes():
    """CUDA类型定义"""
    return ["float32", "float32"]

def get_cuda_torch_inputs(params: Params):
    """生成单个测试用例的输入数据"""
    shape = params.test_shapes[0]

    # Generate hidden states and residual
    hidden_states = torch.randn(shape, dtype=torch.float32, device='cuda') * 0.1
    residual = torch.randn(shape, dtype=torch.float32, device='cuda') * 0.1
    weight = torch.randn(shape[-1], dtype=torch.float32, device='cuda') * 0.1

    return [hidden_states, residual, weight]

def get_all_cuda_torch_inputs(params: Params) -> List[Tuple[Tuple[int, ...], Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]]:
    """生成所有测试形状的输入数据"""
    test_cases = []

    for i, shape in enumerate(params.test_shapes):
        # Generate test data with reasonable ranges
        hidden_states = torch.randn(shape, dtype=torch.float32, device='cuda') * 0.5
        residual = torch.randn(shape, dtype=torch.float32, device='cuda') * 0.5
        weight = torch.randn(shape[-1], dtype=torch.float32, device='cuda') * 0.5

        # Clamp to prevent extreme values
        hidden_states = torch.clamp(hidden_states, -2.0, 2.0)
        residual = torch.clamp(residual, -2.0, 2.0)
        weight = torch.clamp(weight, -1.0, 1.0)

        cuda_inputs = [hidden_states, residual, weight]
        torch_inputs = [hidden_states.clone(), residual.clone(), weight.clone()]

        # Create output tensors
        cuda_output_tensors = []

        test_cases.append((shape, (cuda_inputs, torch_inputs, cuda_output_tensors)))

    return test_cases

def torch_kernel(hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch参考实现：Fused Add RMS Norm
    1. hidden_states = residual + hidden_states
    2. residual = hidden_states
    3. hidden_states = rmsnorm(hidden_states)
    """
    # Step 1: Add residual
    hidden_states = hidden_states + residual

    # Step 2: Update residual
    residual = hidden_states.clone()

    # Step 3: RMS Norm
    # RMS: sqrt(mean(square(x)))
    rms = torch.rsqrt(torch.mean(hidden_states.pow(2), dim=-1, keepdim=True) + eps)

    # Normalize and apply weight
    hidden_states = hidden_states * rms * weight

    return hidden_states, residual

def test_cases():
    """提供测试用例兼容性"""
    params = Params()
    return get_all_cuda_torch_inputs(params)

def torch_fused_add_rms_norm(hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                            eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """兼容性别名"""
    return torch_kernel(hidden_states, residual, weight, eps)