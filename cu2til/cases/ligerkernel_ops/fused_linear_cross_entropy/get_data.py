#!/usr/bin/env python3
"""
Fused Linear Cross Entropy Benchmark - Liger-Kernel Implementation
基于Liger-Kernel的融合线性交叉熵损失实现
"""

import torch
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Params:
    """Fused Linear Cross Entropy 测试参数"""
    test_shapes: List[Tuple[int, int, int]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 覆盖不同模型规模的测试用例 (batch_tokens, hidden_dim, vocab_size)
            self.test_shapes = [
                # Small models
                (512, 512, 1000),     # Small sequence, small hidden, small vocab
                (1024, 768, 2000),    # Medium sequence, medium hidden, medium vocab
                (2048, 1024, 5000),   # Large sequence, large hidden, large vocab

                # Medium models (Llama-7B style)
                (4096, 1024, 32000),  # Llama-7B vocab size
                (8192, 2048, 32000),  # Larger hidden, same vocab
                (16384, 4096, 32000), # Large hidden, full vocab

                # Large models (Llama-70B style)
                (2048, 4096, 50000),  # Large hidden, medium vocab
                (4096, 8192, 50000),  # Very large hidden, large vocab
                (8192, 16384, 50000), # Huge hidden, very large vocab

                # Special cases
                (256, 256, 10000),    # Small sequence, small hidden, medium vocab
                (128, 128, 50000),    # Tiny sequence, tiny hidden, large vocab

                # Vocab size variations
                (1024, 512, 1000),    # Small vocab
                (1024, 512, 10000),   # Medium vocab
                (1024, 512, 50000),   # Large vocab
                (1024, 512, 100000),  # Very large vocab
            ]

def get_cuda_argtypes():
    """CUDA类型定义"""
    return ["float32", "int32"]

def get_cuda_torch_inputs(params: Params):
    """生成单个测试用例的输入数据"""
    batch_tokens, hidden_dim, vocab_size = params.test_shapes[0]

    # Generate input, weight, target
    _input = torch.randn(batch_tokens, hidden_dim, dtype=torch.float32, device='cuda')
    weight = torch.randn(vocab_size, hidden_dim, dtype=torch.float32, device='cuda')
    target = torch.randint(0, vocab_size, (batch_tokens,), dtype=torch.int32, device='cuda')

    return [_input, weight, target]

def get_all_cuda_torch_inputs(params: Params) -> List[Tuple[Tuple[int, ...], Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]]:
    """生成所有测试形状的输入数据"""
    test_cases = []

    for i, shape in enumerate(params.test_shapes):
        batch_tokens, hidden_dim, vocab_size = shape

        # Generate test data with reasonable ranges
        _input = torch.randn(batch_tokens, hidden_dim, dtype=torch.float32, device='cuda') * 0.1
        weight = torch.randn(vocab_size, hidden_dim, dtype=torch.float32, device='cuda') * 0.02
        target = torch.randint(0, vocab_size, (batch_tokens,), dtype=torch.int32, device='cuda')

        # Clamp weight to prevent explosion
        weight = torch.clamp(weight, -0.1, 0.1)

        cuda_inputs = [_input, weight, target]
        torch_inputs = [_input.clone(), weight.clone(), target.clone()]

        # Create output tensors
        cuda_output_tensors = []

        test_cases.append((shape, (cuda_inputs, torch_inputs, cuda_output_tensors)))

    return test_cases

def torch_kernel(_input: torch.Tensor, weight: torch.Tensor, target: torch.Tensor,
                bias: torch.Tensor = None, ignore_index: int = -100, reduction: str = "mean") -> torch.Tensor:
    """
    PyTorch参考实现：Fused Linear Cross Entropy
    先计算线性变换，然后计算交叉熵损失
    """
    # Linear forward: logits = _input @ weight.t()
    logits = torch.matmul(_input, weight.t())
    if bias is not None:
        logits = logits + bias

    # Compute cross entropy loss
    loss = torch.nn.functional.cross_entropy(
        logits, target,
        ignore_index=ignore_index,
        reduction=reduction
    )

    return loss

def test_cases():
    """提供测试用例兼容性"""
    params = Params()
    return get_all_cuda_torch_inputs(params)

def torch_fused_linear_cross_entropy(_input: torch.Tensor, weight: torch.Tensor, target: torch.Tensor,
                                   bias: torch.Tensor = None, ignore_index: int = -100, reduction: str = "mean") -> torch.Tensor:
    """兼容性别名"""
    return torch_kernel(_input, weight, target, bias, ignore_index, reduction)