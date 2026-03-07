import torch
import ctypes
import os
import sys
from dataclasses import dataclass
from torch.nn import functional as F
from cu2til.tools.builder import SEED, load_cuda_kernel
from copy import deepcopy
from cu2til.tools.layout import convert_nhwc_to_nchw, convert_nchw_to_nhwc
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
    
    input_nchw = convert_nhwc_to_nchw(input_tensor)  # NHWC -> NCHW
    kernel_oihw = convert_nhwc_to_nchw(kernel_tensor)  # OHWI -> OIHW
    torch_all_inputs = [input_nchw, kernel_oihw]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def get_cuda_triton_inputs(params: Params):
    """当需要cuda和triton比较时候使用"""
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
    output_triton_nhwc = torch.empty(params.batch_size, params.output_height, params.output_width, params.output_channels, dtype=torch.float32, device="cuda")
    cuda_output_tensors = [output_cuda_nhwc]
    triton_output_tensors = [output_triton_nhwc]
    
    cuda_all_inputs = [input_tensor, kernel_tensor, output_cuda_nhwc, params.batch_size, params.input_height, params.input_channels, params.output_channels, params.kernel_height, params.stride]
    triton_all_inputs = [input_tensor, kernel_tensor, output_triton_nhwc, params.batch_size, params.input_height, params.input_channels, params.output_channels, params.kernel_height, params.stride]
    return cuda_all_inputs, triton_all_inputs, cuda_output_tensors, triton_output_tensors

def get_cuda_triton_torch_inputs(params: Params):
    """当需要cuda和triton比较时候使用"""
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
    output_triton_nhwc = torch.empty(params.batch_size, params.output_height, params.output_width, params.output_channels, dtype=torch.float32, device="cuda")
    cuda_output_tensors = [output_cuda_nhwc]
    triton_output_tensors = [output_triton_nhwc]
    
    cuda_all_inputs = [input_tensor, kernel_tensor, output_cuda_nhwc, params.batch_size, params.input_height, params.input_channels, params.output_channels, params.kernel_height, params.stride]
    triton_all_inputs = [input_tensor, kernel_tensor, output_triton_nhwc, params.batch_size, params.input_height, params.input_channels, params.output_channels, params.kernel_height, params.stride]
    input_nchw = convert_nhwc_to_nchw(input_tensor)  # NHWC -> NCHW
    kernel_oihw = convert_nhwc_to_nchw(kernel_tensor)  # OHWI -> OIHW
    torch_all_inputs = [input_nchw, kernel_oihw]
    return cuda_all_inputs, triton_all_inputs, torch_all_inputs, cuda_output_tensors, triton_output_tensors

def cuda_output_tensor_transform(cuda_output):
    return convert_nhwc_to_nchw(cuda_output)  # NHWC -> NCHW
