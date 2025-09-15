import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Half precision GEMM MMA parameters for dynamic shapes"""
    test_shapes: List[Tuple[int, int, int]] = None  # List of (M, K, N) shapes to test
    
    def __post_init__(self):
        if self.test_shapes is None:
            # Default test shapes covering various (M, K, N) combinations
            self.test_shapes = [(size, size, size) for size in range(256, 4096 + 1, 128)]

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # A (GPU pointer)
        ctypes.c_void_p,  # B (GPU pointer)  
        ctypes.c_void_p,  # C (GPU pointer)
        ctypes.c_int,     # M
        ctypes.c_int,     # K
        ctypes.c_int      # N
    ]

def get_cuda_torch_inputs_for_config(config: Tuple[int, int, int]):
    """Create input data for both CUDA and PyTorch implementations for a specific config"""
    torch.manual_seed(SEED)
    
    M, K, N = config
    
    # Create input matrices
    A = torch.randn(M, K, dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
    B = torch.randn(K, N, dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
    
    # Create output matrix
    C_cuda = torch.empty(M, N, dtype=torch.float16, device="cuda")
    
    # CUDA inputs: GPU pointers list
    cuda_all_inputs = [
        A,
        B,
        C_cuda,
        M,
        N,  # 正确：参考实现期待的参数顺序是M,N,K
        K
    ]
    
    # PyTorch inputs: matrix list for reference implementation
    torch_all_inputs = [A, B]
    
    # CUDA output tensor for result comparison
    cuda_output_tensors = [C_cuda]
    
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def get_all_cuda_torch_inputs(params: Params):
    """Create input data for all test shapes"""
    all_test_data = []
    
    for config in params.test_shapes:
        test_data = get_cuda_torch_inputs_for_config(config)
        all_test_data.append((config, test_data))
    
    return all_test_data

def cuda_output_tensor_transform(cuda_output):
    return cuda_output
