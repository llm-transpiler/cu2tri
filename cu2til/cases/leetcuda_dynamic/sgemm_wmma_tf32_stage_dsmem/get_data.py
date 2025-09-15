import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """sgemm_wmma_tf32_stage parameters for dynamic shapes"""
    test_shapes: List[Tuple[int, int, int]] = None  # List of (M, K, N) shapes to test
    
    def __post_init__(self):
        if self.test_shapes is None:
            # Default test shapes covering various (M, K, N) combinations
            self.test_shapes = []
            self.test_shapes = [(i * 256, i * 256, i * 256) for i in range(2, 17)]
            self.test_shapes.extend([
                (4096, 8192, 2048),
                (8192, 4096, 2048)
            ])

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
    A = torch.randn(M, K, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    B = torch.randn(K, N, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    
    # Create output matrix
    C_cuda = torch.empty(M, N, dtype=torch.float32, device="cuda")
    
    # CUDA inputs: GPU pointers list
    cuda_all_inputs = [
        A,
        B,
        C_cuda,
        M,
        N,  # 保持：sgemm系列确实是M,N,K
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
