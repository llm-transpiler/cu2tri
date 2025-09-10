import torch
import ctypes
import os
import sys
from dataclasses import dataclass
from torch.nn import functional as F
from cu2til.tools.builder import SEED, load_cuda_kernel

@dataclass
class Params:
    """Conv2D 参数配置"""
    batch_size: int = 32
    input_height: int = 8
    input_width: int = 8
    input_channels: int = 128
    output_channels: int = 64
    kernel_height: int = 2
    kernel_width: int = 2
    stride: int = 3
    padding: int = 0
    output_height: int = 3
    output_width: int = 3

def get_cuda_argtypes():
    return [
            ctypes.c_void_p,  # input (GPU pointer)
            ctypes.c_void_p,  # kernel (GPU pointer)
            ctypes.c_void_p,  # output (GPU pointer)
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # input_height
            ctypes.c_int,     # input_channels
            ctypes.c_int,     # output_channels
            ctypes.c_int,     # kernel_height
            ctypes.c_int      # stride
        ]

def get_cuda_inputs(params: Params):
    """当只需要cuda的inputs的时候使用"""
    torch.manual_seed(SEED)
    # Conv2D: conv2d_16_8_8_64_64_2_2_64_2_0
    # 以CUDA kernel的格式需求为准
    # Input: NHWC = (batch_size, input_height, input_width, input_channels) 
    # Kernel: OHWI = (output_channels, kernel_height, kernel_width, input_channels)
    
    # Create data directly on specified device (NHWC and OHWI format)
    input_tensor = torch.randn(params.batch_size, params.input_height, params.input_width, params.input_channels, 
                              dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    kernel_tensor = torch.randn(params.output_channels, params.kernel_height, params.kernel_width, params.input_channels, 
                               dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    
    # Create output tensor (NHWC format for CUDA)
    output_cuda_nhwc = torch.empty(params.batch_size, params.output_height, params.output_width, params.output_channels, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers (input已经是NHWC格式，无需转换)
    input_ptr = input_tensor.data_ptr()
    kernel_ptr = kernel_tensor.data_ptr()
    output_ptr = output_cuda_nhwc.data_ptr()
    
    # Call CUDA kernel
    cuda_all_inputs = [input_ptr, kernel_ptr, output_ptr, params.batch_size, params.input_height, params.input_channels, params.output_channels, params.kernel_height, params.stride]
    return cuda_all_inputs

def get_cuda_torch_inputs(params: Params):
    """当需要cuda和torch比较时候使用"""
    torch.manual_seed(SEED)
    # Conv2D: conv2d_16_8_8_64_64_2_2_64_2_0
    # 以CUDA kernel的格式需求为准
    # Input: NHWC = (batch_size, input_height, input_width, input_channels) 
    # Kernel: OHWI = (output_channels, kernel_height, kernel_width, input_channels)
    
    # Create data directly on specified device (NHWC and OHWI format)
    input_tensor = torch.randn(params.batch_size, params.input_height, params.input_width, params.input_channels, 
                              dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    kernel_tensor = torch.randn(params.output_channels, params.kernel_height, params.kernel_width, params.input_channels, 
                               dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    
    # Create output tensor (NHWC format for CUDA)
    output_cuda_nhwc = torch.empty(params.batch_size, params.output_height, params.output_width, params.output_channels, dtype=torch.float32, device="cuda")
    cuda_output_tensors = [output_cuda_nhwc]
    
    cuda_all_inputs = [input_tensor, kernel_tensor, output_cuda_nhwc, params.batch_size, params.input_height, params.input_channels, params.output_channels, params.kernel_height, params.stride]
    
    input_nchw = input_tensor.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    kernel_oihw = kernel_tensor.permute(0, 3, 1, 2).contiguous()  # OHWI -> OIHW
    torch_all_inputs = [input_nchw, kernel_oihw]
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
    return cuda_output.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
