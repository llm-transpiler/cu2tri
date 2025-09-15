import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Layer Norm f16x8_pack_f32 parameters for dynamic shapes"""
    test_shapes: List[Tuple[int, int]] = None  # List of (N, K) tensor shapes to test
    g: float = 1.0  # gamma (scale)
    b: float = 0.0  # beta (bias)
    
    def __post_init__(self):
        if self.test_shapes is None:
            self.test_shapes = [
                (n, k)
                for n in [512, 1024, 2048]
                for k in [512, 1024, 2048]
            ]

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # x (GPU pointer)
        ctypes.c_void_p,  # y (GPU pointer)
        ctypes.c_float,   # g (gamma)
        ctypes.c_float,   # b (beta)
        ctypes.c_int,     # N
        ctypes.c_int      # K
    ]

def get_cuda_torch_inputs_for_shape(shape: Tuple[int, int], g: float, b: float):
    """Create input data for both CUDA and PyTorch implementations for a specific shape"""
    torch.manual_seed(SEED)
    
    N, K = shape
    
    # Create input tensor (half precision)
    x = torch.randn(shape, dtype=torch.float16, device="cuda").normal_(mean=0.0, std=1.0)
    
    # Create output tensor (float32 for _f32 variant)
    y_cuda = torch.empty(shape, dtype=torch.float16, device="cuda")
    
    # CUDA inputs: GPU pointers list
    cuda_all_inputs = [
        x,
        y_cuda,
        g,
        b,
        N,
        K
    ]
    
    # PyTorch inputs: tensors and parameters for reference implementation
    torch_all_inputs = [x, g, b]
    
    # CUDA output tensor for result comparison
    cuda_output_tensors = [y_cuda]
    
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def get_all_cuda_torch_inputs(params: Params):
    """Create input data for all test shapes"""
    all_test_data = []
    
    for shape in params.test_shapes:
        test_data = get_cuda_torch_inputs_for_shape(shape, params.g, params.b)
        all_test_data.append((shape, test_data))
    
    return all_test_data

def cuda_output_tensor_transform(cuda_output):
    return cuda_output
