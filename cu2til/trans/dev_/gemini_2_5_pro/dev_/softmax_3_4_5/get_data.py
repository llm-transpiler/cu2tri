import torch
import ctypes
import os
import sys
from dataclasses import dataclass
from cu2til.tools.builder import SEED, load_cuda_kernel

@dataclass
class Params:
    """Softmax 参数配置"""
    shape: tuple = (3, 4, 5)
    total_elements: int = 60  # 3 * 4 * 5
    dim: int = -1  # last dimension for softmax

def get_cuda_argtypes():
    return [
            ctypes.c_void_p,  # x (GPU pointer)
            ctypes.c_void_p,  # output (GPU pointer)
            ctypes.c_int,     # batch_size (3)
            ctypes.c_int,     # seq_len (4) 
            ctypes.c_int      # feature_dim (5) - softmax dimension
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
    
    batch_size, seq_len, feature_dim = params.shape
    cuda_all_inputs = [x_ptr, output_ptr, batch_size, seq_len, feature_dim]
    return cuda_all_inputs

def get_cuda_torch_inputs(params: Params):
    """当需要cuda和torch比较时候使用"""
    torch.manual_seed(SEED)
    # Create data directly on specified device
    x = torch.randn(params.shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    
    # Create output tensor
    output_cuda = torch.empty_like(x)
    cuda_output_tensors = [output_cuda]
    
    batch_size, seq_len, feature_dim = params.shape
    cuda_all_inputs = [x, output_cuda, batch_size, seq_len, feature_dim]
    
    # For PyTorch, inputs are already in correct format
    torch_all_inputs = [x]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def cuda_output_tensor_transform(cuda_output):
    # For Softmax operation, no format transformation needed
    # Both CUDA and PyTorch use the same tensor format
    return cuda_output
