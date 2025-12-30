#!/usr/bin/env python3
"""
RoPE Benchmark - Liger-Kernel Implementation
基于Liger-Kernel的旋转位置编码实现
"""

import torch
import math
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Params:
    """RoPE 测试参数"""
    test_shapes: List[Tuple[int, int, int, int]] = None  # (batch_size, seq_len, num_heads, head_dim)

    def __post_init__(self):
        if self.test_shapes is None:
            # 覆盖不同模型规模的测试用例
            self.test_shapes = [
                # Small models
                (2, 128, 8, 64),      # Small batch, short seq, 8 heads, 64 dim
                (4, 256, 8, 128),     # Medium batch, medium seq, 8 heads, 128 dim
                (8, 512, 16, 128),    # Large batch, long seq, 16 heads, 128 dim

                # Medium models (Llama-7B style)
                (4, 1024, 32, 128),   # Small batch, long seq, 32 heads, 128 dim
                (2, 2048, 32, 128),   # Small batch, very long seq, 32 heads, 128 dim
                (1, 4096, 32, 128),   # Single batch, massive seq, 32 heads, 128 dim

                # Large models (Llama-70B style)
                (2, 512, 64, 128),    # Small batch, medium seq, 64 heads, 128 dim
                (1, 1024, 64, 128),   # Single batch, long seq, 64 heads, 128 dim
                (1, 2048, 64, 128),   # Single batch, very long seq, 64 heads, 128 dim

                # Different head dimensions
                (4, 256, 16, 64),     # Small head dim
                (4, 256, 16, 256),    # Large head dim

                # Grouped query attention (GQA) patterns
                (4, 512, 32, 128, 8), # 32 query heads, 8 key heads
                (2, 1024, 64, 128, 8), # 64 query heads, 8 key heads

                # Edge cases
                (1, 1, 1, 64),        # Minimal case
                (16, 64, 4, 32),     # Large batch, tiny seq
            ]

def get_cuda_argtypes():
    """CUDA类型定义"""
    return ["float32", "float32"]

def get_cuda_torch_inputs(params: Params):
    """生成单个测试用例的输入数据"""
    shape = params.test_shapes[0]
    if len(shape) == 4:
        batch_size, seq_len, num_heads, head_dim = shape
        num_kv_heads = num_heads
    else:
        batch_size, seq_len, num_heads, head_dim, num_kv_heads = shape

    # Generate Q and K tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device='cuda')
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=torch.float32, device='cuda')

    return [q, k]

def get_all_cuda_torch_inputs(params: Params) -> List[Tuple[Tuple[int, ...], Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]]:
    """生成所有测试形状的输入数据"""
    test_cases = []

    for i, shape in enumerate(params.test_shapes):
        if len(shape) == 4:
            batch_size, seq_len, num_heads, head_dim = shape
            num_kv_heads = num_heads
        else:
            batch_size, seq_len, num_heads, head_dim, num_kv_heads = shape

        # Generate test data with reasonable ranges
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device='cuda') * 0.1
        k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=torch.float32, device='cuda') * 0.1

        # Clamp to prevent extreme values
        q = torch.clamp(q, -1.0, 1.0)
        k = torch.clamp(k, -1.0, 1.0)

        cuda_inputs = [q, k]
        torch_inputs = [q.clone(), k.clone()]

        # Create output tensors
        cuda_output_tensors = []

        # Store both the shape and num_kv_heads for GQA cases
        full_shape = (batch_size, seq_len, num_heads, head_dim, num_kv_heads) if num_kv_heads != num_heads else shape
        test_cases.append((full_shape, (cuda_inputs, torch_inputs, cuda_output_tensors)))

    return test_cases

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute frequency tensor for complex exponentials (cos, sin)"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors"""
    # Reshape for broadcasting
    head_dim = xq.shape[-1]
    xq_cos = xq * freqs_cos.unsqueeze(-2)
    xq_sin = xq * freqs_sin.unsqueeze(-2)

    # Split into two halves
    xq1, xq2 = xq_cos[..., :head_dim//2], xq_cos[..., head_dim//2:]
    xq3, xq4 = xq_sin[..., :head_dim//2], xq_sin[..., head_dim//2:]

    # Apply rotation
    xq_new = torch.cat([-xq4, xq3], dim=-1) + torch.cat([xq1, xq2], dim=-1)

    # Same for keys
    xk_cos = xk * freqs_cos.unsqueeze(-2)
    xk_sin = xk * freqs_sin.unsqueeze(-2)

    xk1, xk2 = xk_cos[..., :head_dim//2], xk_cos[..., head_dim//2:]
    xk3, xk4 = xk_sin[..., :head_dim//2], xk_sin[..., head_dim//2:]

    xk_new = torch.cat([-xk4, xk3], dim=-1) + torch.cat([xk1, xk2], dim=-1)

    return xq_new, xk_new

def torch_kernel(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch参考实现：RoPE (Rotary Position Embedding)
    """
    batch_size, seq_len, num_heads, head_dim = q.shape

    # Precompute cos and sin values
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len, device=q.device)

    # Apply rotary embeddings
    q_new, k_new = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

    return q_new, k_new

def test_cases():
    """提供测试用例兼容性"""
    params = Params()
    return get_all_cuda_torch_inputs(params)

def torch_rope(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """兼容性别名"""
    return torch_kernel(q, k)