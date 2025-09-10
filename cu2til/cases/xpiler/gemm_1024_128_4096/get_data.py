import torch
import ctypes
from dataclasses import dataclass
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """GEMM (General Matrix Multiplication) operation parameters"""
    M: int = 1024  # A.shape = (M, K)
    K: int = 128  # B.shape = (K, N) 
    N: int = 4096  # C.shape = (M, N)

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # A (half* GPU pointer)
        ctypes.c_void_p,  # B (half* GPU pointer)
        ctypes.c_void_p,  # C (float* GPU pointer)
        ctypes.c_int,     # M
        ctypes.c_int,     # K
        ctypes.c_int      # N
    ]

def get_cuda_inputs(params: Params):
    torch.manual_seed(SEED)
    A = torch.randn((params.M, params.K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
    B = torch.randn((params.K, params.N), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
    C = torch.empty((params.M, params.N), dtype=torch.float32, device="cuda")
    
    cuda_all_inputs = [
        A.data_ptr(), 
        B.data_ptr(), 
        C.data_ptr(),
        params.M,
        params.K,
        params.N
    ]
    return cuda_all_inputs

def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    A_half = torch.randn((params.M, params.K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
    B_half = torch.randn((params.K, params.N), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
    C = torch.empty((params.M, params.N), dtype=torch.float32, device="cuda")
    
    # Convert to float32 for PyTorch comparison
    A_float32 = A_half.float()
    B_float32 = B_half.float()
    
    cuda_output_tensors = [C]
    
    cuda_all_inputs = [
        A_half, B_half, C,
        params.M, params.K, params.N
    ]
    torch_all_inputs = [A_float32, B_float32]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def cuda_input_tensor_to_ptr(cuda_all_inputs):
    return [t.data_ptr() if isinstance(t, torch.Tensor) else t for t in cuda_all_inputs]

def cuda_output_tensor_transform(cuda_output):
    return cuda_output  # No transformation needed for GEMM
