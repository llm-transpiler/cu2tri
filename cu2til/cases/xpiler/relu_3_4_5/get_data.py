import torch
import ctypes
import os
import sys
from dataclasses import dataclass
from cu2til.tools.builder import SEED, load_cuda_kernel

@dataclass
class Params:
    """ReLU 参数配置"""
    shape: tuple = (3, 4, 5)
    total_elements: int = 60  # 3 * 4 * 5

def get_cuda_argtypes():
    return [
            ctypes.c_void_p,  # x (GPU pointer)
            ctypes.c_void_p,  # output (GPU pointer)
            ctypes.c_int      # size
        ]

def get_cuda_inputs(params: Params):
    """当只需要cuda的inputs的时候使用"""
    torch.manual_seed(SEED)
    # Create data directly on specified device
    x = torch.randn(params.shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    
    # Create output tensor
    output_cuda = torch.empty_like(x)
    
    # Get GPU pointers
    x_ptr = x.data_ptr()
    output_ptr = output_cuda.data_ptr()
    
    cuda_all_inputs = [x_ptr, output_ptr, params.total_elements]
    return cuda_all_inputs

def get_cuda_torch_inputs(params: Params):
    """当需要cuda和torch比较时候使用"""
    torch.manual_seed(SEED)
    # Create data directly on specified device
    x = torch.randn(params.shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    
    # Create output tensor
    output_cuda = torch.empty_like(x)
    cuda_output_tensors = [output_cuda]
    
    cuda_all_inputs = [x, output_cuda, params.total_elements]
    
    # For PyTorch, inputs are already in correct format
    torch_all_inputs = [x]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def cuda_input_tensor_to_ptr(cuda_all_inputs):
    cuda_all_inputs_ptr = []
    for input_tensor in cuda_all_inputs:
        if isinstance(input_tensor, torch.Tensor):
            cuda_all_inputs_ptr.append(input_tensor.data_ptr())
        else:
            cuda_all_inputs_ptr.append(input_tensor)
    return cuda_all_inputs_ptr

def cuda_output_tensor_transform(cuda_output):
    # For ReLU operation, no format transformation needed
    # Both CUDA and PyTorch use the same tensor format
    return cuda_output
