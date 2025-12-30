#!/usr/bin/env python3
"""
GRPO Loss Benchmark - Liger-Kernel Implementation
基于Liger-Kernel的GRPO损失函数实现
"""

import torch
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Params:
    """GRPO Loss 测试参数"""
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
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
    return ["float32", "float32"]

def get_cuda_torch_inputs(params: Params):
    shape = params.test_shapes[0]

    # Generate logits and advantages
    logits = torch.randn(shape, dtype=torch.float32, device='cuda') - 2.0
    advantages = torch.randn(shape[:-1], dtype=torch.float32, device='cuda')

    return [logits, advantages]

def get_all_cuda_torch_inputs(params: Params):
    test_cases = []

    for shape in params.test_shapes:
        logits = torch.randn(shape, dtype=torch.float32, device='cuda') - 2.0
        advantages = torch.randn(shape[:-1], dtype=torch.float32, device='cuda') * 0.5

        logits = torch.clamp(logits, -10.0, 5.0)
        advantages = torch.clamp(advantages, -2.0, 2.0)

        cuda_inputs = [logits, advantages]
        torch_inputs = [logits.clone(), advantages.clone()]

        test_cases.append((shape, (cuda_inputs, torch_inputs, [])))

    return test_cases

def torch_kernel(logits: torch.Tensor, advantages: torch.Tensor, epsilon: float = 0.2) -> torch.Tensor:
    """
    PyTorch参考实现：GRPO (Generalized Relative Policy Optimization) Loss
    """
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)

    # Compute log_probs
    log_probs = torch.log_softmax(logits, dim=-1)

    # Sample action (use argmax for deterministic reference)
    actions = torch.argmax(probs, dim=-1)

    # Get action probabilities
    batch_indices = torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1)
    if len(logits.shape) > 2:
        seq_indices = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0).unsqueeze(-1)
        batch_indices = batch_indices.unsqueeze(1).expand(-1, logits.shape[1], -1)
        seq_indices = seq_indices.expand_as(batch_indices)
        action_indices = torch.stack([batch_indices, seq_indices, actions], dim=-1)
    else:
        action_indices = torch.stack([batch_indices, actions], dim=-1)

    # For simplicity in reference, just use max probability
    action_probs = torch.max(probs, dim=-1)[0]
    action_log_probs = torch.log(action_probs + 1e-8)

    # GRPO surrogate objective
    ratio = torch.exp(action_log_probs - action_log_probs.detach())  # Simplified
    surrogate = ratio * advantages

    # Clipped surrogate
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    clipped_surrogate = clipped_ratio * advantages

    # GRPO loss: minimize negative of clipped objective
    loss = -torch.minimum(surrogate, clipped_surrogate).mean()

    return loss

def test_cases():
    params = Params()
    return get_all_cuda_torch_inputs(params)

def torch_grpo_loss(logits: torch.Tensor, advantages: torch.Tensor, epsilon: float = 0.2) -> torch.Tensor:
    return torch_kernel(logits, advantages, epsilon)