import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Elementwise Add f32x4 parameters for dynamic shapes"""
    test_shapes: List[Tuple[int, ...]] = None  # List of tensor shapes to test
    
    def __post_init__(self):
        if self.test_shapes is None:
            # Default test shapes covering various sizes
            self.test_shapes = [
                (256,),     # Small
                (512,),     # Medium small  
                (1024,),    # Medium
                (2048,),    # Medium large
                (4096,),    # Large
                (8192,),    # Very large
                (16384,),   # Extra large
                (32768,),   # Huge
            ]

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # a (GPU pointer)
        ctypes.c_void_p,  # b (GPU pointer)
        ctypes.c_void_p,  # c (GPU pointer)
        ctypes.c_int      # N
    ]

def get_cuda_torch_inputs_for_shape(shape: Tuple[int, ...]):
    """Create input data for both CUDA and PyTorch implementations for a specific shape"""
    torch.manual_seed(SEED)
    
    N = shape[0] if len(shape) == 1 else torch.prod(torch.tensor(shape)).item()
    
    # Create input tensors with the given shape
    a = torch.randn(shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    b = torch.randn(shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    
    # Create output tensor
    c_cuda = torch.empty_like(a)
    
    # CUDA inputs: GPU pointers list
    cuda_all_inputs = [
        a,
        b,
        c_cuda,
        N
    ]
    
    # PyTorch inputs: tensor list for reference implementation
    torch_all_inputs = [a, b]
    
    # CUDA output tensor for result comparison
    cuda_output_tensors = [c_cuda]
    
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def get_all_cuda_torch_inputs(params: Params):
    """Create input data for all test shapes"""
    all_test_data = []
    
    for shape in params.test_shapes:
        test_data = get_cuda_torch_inputs_for_shape(shape)
        all_test_data.append((shape, test_data))
    
    return all_test_data

def cuda_output_tensor_transform(cuda_output):
    return cuda_output
