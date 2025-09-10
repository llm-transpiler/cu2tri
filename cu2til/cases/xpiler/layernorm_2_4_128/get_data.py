import torch
import ctypes
from dataclasses import dataclass
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """LAYERNORM operation parameters"""
    shape: tuple = (2, 4, 128)
    d_model: int = 128
    total_elements: int = 1024

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,     # total_elements or batch_size
        ctypes.c_int      # d_model
    ]

def get_cuda_inputs(params: Params):
    torch.manual_seed(SEED)
    input_tensor = torch.randn(params.shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    output_tensor = torch.empty_like(input_tensor)
    
    cuda_all_inputs = [
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        params.total_elements,
        params.d_model
    ]
    return cuda_all_inputs

def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    input_tensor = torch.randn(params.shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    output_tensor = torch.empty_like(input_tensor)
    
    cuda_output_tensors = [output_tensor]
    
    cuda_all_inputs = [
        input_tensor,
        output_tensor,
        params.total_elements,
        params.d_model
    ]
    torch_all_inputs = [input_tensor]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def cuda_input_tensor_to_ptr(cuda_all_inputs):
    return [t.data_ptr() if isinstance(t, torch.Tensor) else t for t in cuda_all_inputs]

def cuda_output_tensor_transform(cuda_output):
    return cuda_output  # No transformation needed for LAYERNORM
