import torch
import ctypes
from dataclasses import dataclass
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Deformable Convolution operation parameters"""
    batch_size: int = 4
    channels: int = 8
    input_dim: int = 256
    output_dim: int = 200
    kernel_h: int = 4
    kernel_w: int = 4

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input (GPU pointer)
        ctypes.c_void_p,  # offset (GPU pointer)
        ctypes.c_void_p,  # weight (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,     # batch_size
        ctypes.c_int,     # channels
        ctypes.c_int,     # input_dim
        ctypes.c_int,     # output_dim
        ctypes.c_int,     # kernel_h
        ctypes.c_int      # kernel_w
    ]

def get_cuda_inputs(params: Params):
    torch.manual_seed(SEED)
    input_tensor = torch.randn((params.batch_size, params.channels, params.input_dim, params.input_dim), dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    offset_tensor = torch.randn((params.batch_size, 2*params.kernel_h*params.kernel_w, params.output_dim, params.output_dim), dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.1)
    weight_tensor = torch.randn((params.channels, params.channels, params.kernel_h, params.kernel_w), dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.1)
    output_tensor = torch.empty((params.batch_size, params.channels, params.output_dim, params.output_dim), dtype=torch.float32, device="cuda")
    
    cuda_all_inputs = [input_tensor.data_ptr(), offset_tensor.data_ptr(), weight_tensor.data_ptr(), output_tensor.data_ptr(), params.batch_size, params.channels, params.input_dim, params.output_dim, params.kernel_h, params.kernel_w]
    return cuda_all_inputs

def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    input_tensor = torch.randn((params.batch_size, params.channels, params.input_dim, params.input_dim), dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    offset_tensor = torch.randn((params.batch_size, 2*params.kernel_h*params.kernel_w, params.output_dim, params.output_dim), dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.1)
    weight_tensor = torch.randn((params.channels, params.channels, params.kernel_h, params.kernel_w), dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.1)
    output_tensor = torch.empty((params.batch_size, params.channels, params.output_dim, params.output_dim), dtype=torch.float32, device="cuda")
    
    cuda_output_tensors = [output_tensor]
    cuda_all_inputs = [input_tensor, offset_tensor, weight_tensor, output_tensor, params.batch_size, params.channels, params.input_dim, params.output_dim, params.kernel_h, params.kernel_w]
    torch_all_inputs = [input_tensor, offset_tensor, weight_tensor]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def cuda_input_tensor_to_ptr(cuda_all_inputs):
    return [t.data_ptr() if isinstance(t, torch.Tensor) else t for t in cuda_all_inputs]

def cuda_output_tensor_transform(cuda_output):
    return cuda_output
