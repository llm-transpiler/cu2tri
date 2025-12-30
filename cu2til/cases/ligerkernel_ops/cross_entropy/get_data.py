#!/usr/bin/env python3
"""
Cross Entropy Loss Benchmark - Liger-Kernel Implementation
基于Liger-Kernel的高性能CrossEntropy实现
"""

import torch
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Params:
    """Cross Entropy Loss 测试参数"""
    test_shapes: List[Tuple[int, int, int]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 覆盖不同模型规模的测试用例 (batch_size, seq_len, vocab_size)
            # 适度规模以避免GPU内存不足
            self.test_shapes = [
                # Small models
                (1, 64, 8000),     # Small batch, short sequence, small vocab
                (2, 128, 16000),   # Medium batch, medium sequence, medium vocab
                (1, 256, 32000),   # Larger batch, longer sequence, larger vocab

                # Medium models (Llama-7B style)
                (1, 128, 32000),   # Small batch, medium sequence, medium vocab
                (2, 256, 32000),   # Medium batch, long sequence, medium vocab
                (1, 512, 32000),   # Larger batch, very long sequence, medium vocab

                # Large models (Llama-70B style)
                (1, 64, 64000),    # Small batch, short sequence, large vocab
                (1, 128, 64000),   # Small batch, medium sequence, large vocab
                (1, 256, 64000),   # Medium batch, long sequence, large vocab

                # Special cases
                (2, 32, 16000),    # Very short sequence, medium batch
                (4, 16, 32000),    # Tiny sequence, large batch
                (8, 8, 8000),      # Very tiny sequence, medium batch
            ]

def get_cuda_argtypes():
    """CUDA类型定义"""
    return ["float32", "float32", "float32"]

def get_cuda_torch_inputs(params: Params):
    """生成单个测试用例的输入数据"""
    batch_size, seq_len, vocab_size = params.test_shapes[0]

    # Generate random logits
    logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32, device='cuda')

    # Generate random labels (including padding tokens -100)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
    # Add some padding tokens (10% chance)
    mask = torch.rand((batch_size, seq_len), device='cuda') < 0.1
    labels[mask] = -100

    return [logits, labels]

def get_all_cuda_torch_inputs(params: Params) -> List[Tuple[Tuple[int, ...], Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]]:
    """生成所有测试形状的输入数据"""
    test_cases = []

    for i, shape in enumerate(params.test_shapes):
        batch_size, seq_len, vocab_size = shape

        # Generate test data
        logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32, device='cuda') * 0.5
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')

        # Add padding tokens
        mask = torch.rand((batch_size, seq_len), device='cuda') < 0.1
        labels[mask] = -100

        cuda_inputs = [logits, labels]
        torch_inputs = [logits.clone(), labels.clone()]

        # Create output tensors
        cuda_output_tensors = []

        test_cases.append((shape, (cuda_inputs, torch_inputs, cuda_output_tensors)))

    return test_cases

def torch_kernel(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """PyTorch参考实现：Cross Entropy Loss"""
    batch_size, seq_len, vocab_size = logits.shape

    # Reshape for cross entropy
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Compute cross entropy loss
    loss = torch.nn.functional.cross_entropy(
        logits_flat,
        labels_flat,
        reduction='mean',
        ignore_index=ignore_index
    )

    return loss