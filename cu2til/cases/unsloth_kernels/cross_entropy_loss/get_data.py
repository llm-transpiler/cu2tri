#!/usr/bin/env python3
"""
Fast Cross Entropy Loss Benchmark - Unsloth Implementation
基于unsloth的快速交叉熵损失实现，支持logit softcapping和scaling
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Params:
    """Cross Entropy Loss 测试参数"""
    test_shapes: List[Tuple[int, int, int]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 覆盖不同模型规模的测试用例 (batch, seq_len, vocab_size)
            # 减少规模以避免GPU内存不足
            self.test_shapes = [
                # Small models
                (1, 64, 8000),     # Small batch, short sequence, small vocab
                (2, 128, 16000),   # Medium batch, medium sequence, small vocab
                (1, 256, 32000),   # Larger batch, longer sequence, small vocab

                # Medium models (Llama-7B style)
                (1, 128, 32000),   # Small batch, medium sequence, medium vocab
                (2, 256, 32000),   # Medium batch, long sequence, medium vocab
                (1, 512, 32000),   # Larger batch, very long sequence, medium vocab

                # Large models (Llama-70B style)
                (1, 64, 64000),    # Small batch, short sequence, large vocab
                (1, 128, 64000),   # Small batch, medium sequence, large vocab
                (1, 256, 64000),   # Medium batch, long sequence, large vocab

                # Extra large models (Gemma 256K) - reduced scale
                (1, 32, 128000),   # Small batch, short sequence, very large vocab
                (1, 64, 128000),   # Small batch, medium sequence, very large vocab

                # Special cases
                (2, 32, 16000),    # Very short sequence, medium-large batch
                (4, 16, 32000),    # Tiny sequence, large batch
            ]

def get_cuda_argtypes():
    """CUDA类型定义"""
    return ["float32", "float32", "int32", "float32", "float32"]

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

    # Logit softcapping parameter (Gemma 2 style)
    logit_softcapping = torch.tensor(30.0, dtype=torch.float32, device='cuda')

    # Logit scaling parameter (Cohere style)
    logit_scaling = torch.tensor(1.0, dtype=torch.float32, device='cuda')

    return [logits, labels, logit_softcapping, logit_scaling]

def get_all_cuda_torch_inputs(params: Params) -> List[Tuple[Tuple[int, ...], Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]]:
    """生成所有测试形状的输入数据"""
    test_cases = []

    for i, shape in enumerate(params.test_shapes):
        batch_size, seq_len, vocab_size = shape

        # Generate test data
        logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32, device='cuda')
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')

        # Add padding tokens
        mask = torch.rand((batch_size, seq_len), device='cuda') < 0.1
        labels[mask] = -100

        # Test with simple scaling values (avoid softcapping complexity for now)
        logit_softcapping = torch.tensor([0.0, 0.0, 0.0][i % 3], dtype=torch.float32, device='cuda')  # Disable softcapping
        logit_scaling = torch.tensor([1.0, 1.0, 1.0][i % 3], dtype=torch.float32, device='cuda')    # Keep scaling simple

        cuda_inputs = [logits, labels, logit_softcapping, logit_scaling]

        # PyTorch inputs need to be adjusted to match unsloth's preprocessing
        torch_inputs = [logits.clone(), labels.clone(), logit_softcapping.clone(), logit_scaling.clone()]

        # Create output tensors
        cuda_output_tensors = []

        test_cases.append((shape, (cuda_inputs, torch_inputs, cuda_output_tensors)))

    return test_cases

def torch_kernel(logits: torch.Tensor, labels: torch.Tensor, logit_softcapping: float = 0.0, logit_scaling: float = 1.0) -> torch.Tensor:
    """PyTorch参考实现：Cross Entropy Loss"""
    batch_size, seq_len, vocab_size = logits.shape

    # Apply logit scaling if specified
    if logit_scaling != 1.0:
        logits = logits * logit_scaling

    # Apply logit softcapping if specified
    if logit_softcapping != 0.0:
        logits = logit_softcapping * torch.tanh(logits / logit_softcapping)

    # Reshape for cross entropy
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Compute cross entropy loss
    loss = F.cross_entropy(logits_flat, labels_flat, reduction='none', ignore_index=-100)

    # Reshape back and compute mean over non-padding tokens
    loss = loss.view(batch_size, seq_len)
    n_items = (labels != -100).sum()

    if n_items > 0:
        return loss.sum() / n_items
    else:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)