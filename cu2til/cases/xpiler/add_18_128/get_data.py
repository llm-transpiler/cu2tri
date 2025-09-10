import torch
import ctypes
from dataclasses import dataclass
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Add operation parameters"""
    dim1: int = 18
    dim2: int = 128
    shape: tuple = (18, 128)
    total_elements: int = 2304  # 18 * 128

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # A (GPU pointer)
        ctypes.c_void_p,  # B (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int      # size
    ]

def get_cuda_inputs(params: Params):
    torch.manual_seed(SEED)
    A = torch.randn(params.shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    B = torch.randn(params.shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    output_cuda = torch.empty_like(A)
    cuda_all_inputs = [A.data_ptr(), B.data_ptr(), output_cuda.data_ptr(), params.total_elements]
    return cuda_all_inputs

def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    A = torch.randn(params.shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    B = torch.randn(params.shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    output_cuda = torch.empty_like(A)
    cuda_output_tensors = [output_cuda]
    
    cuda_all_inputs = [A, B, output_cuda, params.total_elements]
    torch_all_inputs = [A, B]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def cuda_input_tensor_to_ptr(cuda_all_inputs):
    return [t.data_ptr() if isinstance(t, torch.Tensor) else t for t in cuda_all_inputs]

def cuda_output_tensor_transform(cuda_output):
    return cuda_output # No transformation needed for Add
