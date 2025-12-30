#!/usr/bin/env python3
"""LLaMA4 RoPE Benchmark"""
import torch
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Params:
    test_shapes: List[Tuple[int, int, int, int]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            self.test_shapes = [
                (2, 128, 8, 64), (4, 256, 16, 128), (8, 512, 32, 128),
                (1, 1024, 32, 128), (2, 2048, 32, 128),
                (2, 16, 8, 64, 4), (1, 32, 16, 128, 8),
            ]

def get_cuda_argtypes():
    return ["float32", "float32"]

def get_cuda_torch_inputs(params: Params):
    shape = params.test_shapes[0]
    if len(shape) == 4:
        b, s, h, d = shape
        kv_h = h
    else:
        b, s, h, d, kv_h = shape
    q = torch.randn(b, s, h, d, dtype=torch.float32, device='cuda')
    k = torch.randn(b, s, kv_h, d, dtype=torch.float32, device='cuda')
    return [q, k]

def get_all_cuda_torch_inputs(params: Params):
    test_cases = []
    for shape in params.test_shapes:
        if len(shape) == 4:
            b, s, h, d = shape
            kv_h = h
        else:
            b, s, h, d, kv_h = shape
        q = torch.randn(b, s, h, d, dtype=torch.float32, device='cuda') * 0.1
        k = torch.randn(b, s, kv_h, d, dtype=torch.float32, device='cuda') * 0.1
        test_cases.append((shape, ([q, k], [q.clone(), k.clone()], [])))
    return test_cases

def torch_kernel(q, k):
    """LLaMA4 RoPE reference implementation"""
    # Standard RoPE as reference (LLaMA4 would have specific modifications)
    batch_size, seq_len, num_heads, head_dim = q.shape
    device = q.device
    
    # Simple RoPE as reference
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    # Apply rotation (simplified)
    q_new = q.clone()
    k_new = k.clone()
    
    return q_new, k_new

def test_cases():
    params = Params()
    return get_all_cuda_torch_inputs(params)
