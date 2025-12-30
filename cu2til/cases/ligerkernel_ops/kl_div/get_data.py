#!/usr/bin/env python3
"""
KL Divergence Loss Benchmark - Liger-Kernel Implementation
基于Liger-Kernel的KL散度损失实现
"""

import torch
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Params:
    """KL Divergence 测试参数"""
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

                # 3D tensors (batch, seq_len, vocab_size)
                (2, 16, 1000),      # Small batch, short seq, small vocab
                (4, 32, 5000),      # Medium batch, medium seq, medium vocab
                (1, 64, 10000),     # Single batch, long seq, large vocab

                # Edge cases
                (1, 10),            # Minimal case
                (128, 100),         # Large batch, tiny vocab
                (256, 256),         # Square dimensions
            ]

def get_cuda_argtypes():
    """CUDA类型定义"""
    return ["float32", "float32"]

def get_cuda_torch_inputs(params: Params):
    """生成单个测试用例的输入数据"""
    shape = params.test_shapes[0]

    # Generate predictions (in log-space) and targets
    predictions = torch.randn(shape, dtype=torch.float32, device='cuda')
    targets = torch.randn(shape, dtype=torch.float32, device='cuda')

    # Ensure targets sum to 1 along last dimension (probability distribution)
    targets = torch.softmax(targets, dim=-1)

    return [predictions, targets]

def get_all_cuda_torch_inputs(params: Params) -> List[Tuple[Tuple[int, ...], Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]]:
    """生成所有测试形状的输入数据"""
    test_cases = []

    for i, shape in enumerate(params.test_shapes):
        # Generate test data with reasonable ranges
        # Predictions in log-space
        predictions = torch.randn(shape, dtype=torch.float32, device='cuda') - 2.0  # Shifted to negative for log-space
        # Targets as probability distributions
        targets = torch.randn(shape, dtype=torch.float32, device='cuda')
        targets = torch.softmax(targets, dim=-1)

        # Clamp predictions to prevent extreme values
        predictions = torch.clamp(predictions, -10.0, 5.0)

        cuda_inputs = [predictions, targets]
        torch_inputs = [predictions.clone(), targets.clone()]

        # Create output tensors
        cuda_output_tensors = []

        test_cases.append((shape, (cuda_inputs, torch_inputs, cuda_output_tensors)))

    return test_cases

def torch_kernel(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = "batchmean") -> torch.Tensor:
    """
    PyTorch参考实现：KL Divergence Loss
    KL(P||Q) = sum(P * (log(P) - log(Q)))

    Args:
        predictions: Predictions in log-space (log probabilities)
        targets: Target probability distributions
        reduction: Reduction method ("none", "batchmean", "sum", "mean")

    Returns:
        KL Divergence loss
    """
    # PyTorch's kl_div expects input in log-space and target in probability space
    loss = torch.nn.functional.kl_div(predictions, targets, reduction=reduction, log_target=False)
    return loss

def test_cases():
    """提供测试用例兼容性"""
    params = Params()
    return get_all_cuda_torch_inputs(params)

def torch_kl_div(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = "batchmean") -> torch.Tensor:
    """兼容性别名"""
    return torch_kernel(predictions, targets, reduction)