import torch
import ctypes
from dataclasses import dataclass
from cu2til.tools.checker import SEED

@dataclass
class Params:
    batch_size: int = 16
    input_height: int = 8
    input_width: int = 8
    input_channels: int = 64
    output_channels: int = 64
    kernel_height: int = 2
    kernel_width: int = 2
    stride: int = 2
    padding: int = 0
    output_height: int = 4
    output_width: int = 4

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

def get_cuda_tensor_inputs(params: Params):
    torch.manual_seed(SEED)
    input_tensor = torch.randn(params.batch_size, params.input_channels, params.input_height, params.input_width, 
                              dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5).to(memory_format=torch.channels_last)
    kernel_tensor = torch.randn(params.output_channels, params.input_channels, params.kernel_height, params.kernel_width, 
                               dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5).to(memory_format=torch.channels_last)
    
    cuda_output = torch.empty(params.batch_size, params.output_channels, params.output_height, params.output_width, dtype=torch.float32, device="cuda").to(memory_format=torch.channels_last)
    
    cuda_all_inputs = [input_tensor, kernel_tensor, cuda_output, params.batch_size, params.input_height, params.input_channels, params.output_channels, params.kernel_height, params.stride]
    cuda_output_tensors = [cuda_output]
    return cuda_all_inputs, cuda_output_tensors

def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    input_tensor = torch.randn(params.batch_size, params.input_channels, params.input_height, params.input_width, 
                              dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5).to(memory_format=torch.channels_last)
    kernel_tensor = torch.randn(params.output_channels, params.input_channels, params.kernel_height, params.kernel_width, 
                               dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5).to(memory_format=torch.channels_last)
    cuda_output = torch.empty(params.batch_size, params.output_channels, params.output_height, params.output_width, dtype=torch.float32, device="cuda").to(memory_format=torch.channels_last)
    
    cuda_all_inputs = [input_tensor, kernel_tensor, cuda_output, params.batch_size, params.input_height, params.input_channels, params.output_channels, params.kernel_height, params.stride]
    torch_all_inputs = [input_tensor, kernel_tensor]
    cuda_output_tensors = [cuda_output]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def get_cuda_triton_inputs(params: Params):
    torch.manual_seed(SEED)
    input_tensor = torch.randn(params.batch_size, params.input_channels, params.input_height, params.input_width, 
                              dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5).to(memory_format=torch.channels_last)
    kernel_tensor = torch.randn(params.output_channels, params.input_channels, params.kernel_height, params.kernel_width, 
                               dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5).to(memory_format=torch.channels_last)

    cuda_output = torch.empty(params.batch_size, params.output_channels, params.output_height, params.output_width, dtype=torch.float32, device="cuda").to(memory_format=torch.channels_last)
    triton_output = torch.empty(params.batch_size, params.output_channels, params.output_height, params.output_width, dtype=torch.float32, device="cuda").to(memory_format=torch.channels_last)
    
    cuda_all_inputs = [input_tensor, kernel_tensor, cuda_output, params.batch_size, params.input_height, params.input_channels, params.output_channels, params.kernel_height, params.stride]
    triton_all_inputs = [input_tensor, kernel_tensor, triton_output, params.batch_size, params.input_height, params.input_channels, params.output_channels, params.kernel_height, params.stride]
    cuda_output_tensors = [cuda_output]
    triton_output_tensors = [triton_output]
    return cuda_all_inputs, triton_all_inputs, cuda_output_tensors, triton_output_tensors

def get_cuda_triton_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    input_tensor = torch.randn(params.batch_size, params.input_channels, params.input_height, params.input_width, 
                              dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5).to(memory_format=torch.channels_last)
    kernel_tensor = torch.randn(params.output_channels, params.input_channels, params.kernel_height, params.kernel_width, 
                               dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5).to(memory_format=torch.channels_last)
    
    cuda_output = torch.empty(params.batch_size, params.output_channels, params.output_height, params.output_width, dtype=torch.float32, device="cuda").to(memory_format=torch.channels_last)
    triton_output = torch.empty(params.batch_size, params.output_channels, params.output_height, params.output_width, dtype=torch.float32, device="cuda").to(memory_format=torch.channels_last)
    cuda_output_tensors = [cuda_output]
    triton_output_tensors = [triton_output]
    
    cuda_all_inputs = [input_tensor, kernel_tensor, cuda_output, params.batch_size, params.input_height, params.input_channels, params.output_channels, params.kernel_height, params.stride]
    triton_all_inputs = [input_tensor, kernel_tensor, triton_output, params.batch_size, params.input_height, params.input_channels, params.output_channels, params.kernel_height, params.stride]
    torch_all_inputs = [input_tensor, kernel_tensor]
    return cuda_all_inputs, triton_all_inputs, torch_all_inputs, cuda_output_tensors, triton_output_tensors

def cuda_input_tensor_to_ptr(cuda_all_inputs):
    return [t.data_ptr() if isinstance(t, torch.Tensor) else t for t in cuda_all_inputs]

