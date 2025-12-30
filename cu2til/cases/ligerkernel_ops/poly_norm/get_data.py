#!/usr/bin/env python3
"""Polynomial Normalization Benchmark"""
import torch
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Params:
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            self.test_shapes = [
                (16, 128), (32, 256), (64, 512),
                (8, 1024), (16, 2048), (32, 4096),
                (4, 4096), (8, 8192), (16, 16384),
                (2, 16, 512), (4, 32, 1024), (1, 64, 2048),
                (1, 32768), (256, 32),
            ]

def get_cuda_argtypes():
    return ["float32"]

def get_cuda_torch_inputs(params: Params):
    shape = params.test_shapes[0]
    x = torch.randn(shape, dtype=torch.float32, device='cuda') * 0.5
    return [x]

def get_all_cuda_torch_inputs(params: Params):
    test_cases = []
    for shape in params.test_shapes:
        x = torch.randn(shape, dtype=torch.float32, device='cuda') * 0.5
        x = torch.clamp(x, -2.0, 2.0)
        test_cases.append((shape, ([x], [x.clone()], [])))
    return test_cases

def torch_kernel(x: torch.Tensor, alpha: float = 1.0, beta: float = 1.0) -> torch.Tensor:
    """Polynomial Normalization: y = (x + beta)^alpha"""
    return torch.clamp(x + beta, min=1e-8) ** alpha

def test_cases():
    params = Params()
    return get_all_cuda_torch_inputs(params)