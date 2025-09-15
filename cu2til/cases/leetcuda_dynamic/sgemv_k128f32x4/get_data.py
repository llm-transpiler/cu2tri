import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """sgemv_k128f32x4 parameters for dynamic shapes"""
    test_shapes: List[Tuple[int, int]] = None  # List of (M, K) shapes to test
    
    def __post_init__(self):
        if self.test_shapes is None:
            # Default test shapes with fixed K=128, varying M
            self.test_shapes = [
                (128, 128),   # Small M
                (256, 128),   # Medium M
                (512, 128),   # Large M
                (1024, 128),  # Very large M
                (2048, 128),  # Extra large M
                (4096, 128),  # Very wide M
                (8192, 128),  # Extremely large M
                (16384, 128), # Max M
            ]

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # A (GPU pointer)
        ctypes.c_void_p,  # x (GPU pointer)
        ctypes.c_void_p,  # y (GPU pointer)
        ctypes.c_int,     # M
        ctypes.c_int      # K
    ]

def get_cuda_torch_inputs_for_config(config: Tuple[int, int]):
    """Create input data for both CUDA and PyTorch implementations for a specific config"""
    torch.manual_seed(SEED)
    
    M, K = config
    
    # Create input matrix and vector
    A = torch.randn(M, K, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    x = torch.randn(K, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    
    # Create output vector
    y_cuda = torch.empty(M, dtype=torch.float32, device="cuda")
    
    # CUDA inputs: GPU pointers list
    cuda_all_inputs = [
        A,
        x,
        y_cuda,
        M,
        K
    ]
    
    # PyTorch inputs: matrix and vector for reference implementation
    torch_all_inputs = [A, x]
    
    # CUDA output tensor for result comparison
    cuda_output_tensors = [y_cuda]
    
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
