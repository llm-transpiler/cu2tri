import torch
import ctypes
from dataclasses import dataclass
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """GEMV (General Matrix-Vector Multiplication) operation parameters"""
    M: int = 125  # A.shape = (M, N)
    N: int = 320  # x.shape = (N,), y.shape = (M,)

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # A (GPU pointer)
        ctypes.c_void_p,  # x (GPU pointer)
        ctypes.c_void_p,  # y (GPU pointer)
        ctypes.c_int,     # M
        ctypes.c_int      # N
    ]

def get_cuda_inputs(params: Params):
    torch.manual_seed(SEED)
    A = torch.randn((params.M, params.N), dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    x = torch.randn((params.N,), dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    y = torch.empty((params.M,), dtype=torch.float32, device="cuda")
    
    cuda_all_inputs = [
        A.data_ptr(), 
        x.data_ptr(), 
        y.data_ptr(),
        params.M,
        params.N
    ]
    return cuda_all_inputs

def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    A = torch.randn((params.M, params.N), dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    x = torch.randn((params.N,), dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    y = torch.empty((params.M,), dtype=torch.float32, device="cuda")
    
    cuda_output_tensors = [y]
    
    cuda_all_inputs = [
        A, x, y,
        params.M, params.N
    ]
    torch_all_inputs = [A, x]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def cuda_input_tensor_to_ptr(cuda_all_inputs):
    return [t.data_ptr() if isinstance(t, torch.Tensor) else t for t in cuda_all_inputs]

def cuda_output_tensor_transform(cuda_output):
    return cuda_output  # No transformation needed for GEMV
