import torch
import ctypes
from dataclasses import dataclass
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """MHA (Multi-Head Attention) operation parameters"""
    batch_size: int = 64
    seq_len: int = 2048
    num_heads: int = 12
    d_head: int = 256

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # Q (GPU pointer)
        ctypes.c_void_p,  # K (GPU pointer)
        ctypes.c_void_p,  # V (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,     # batch_size
        ctypes.c_int,     # seq_len
        ctypes.c_int,     # num_heads
        ctypes.c_int      # d_head
    ]

def get_cuda_inputs(params: Params):
    torch.manual_seed(SEED)
    Q = torch.randn((params.batch_size, params.seq_len, params.num_heads, params.d_head), 
                    dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    K = torch.randn((params.batch_size, params.seq_len, params.num_heads, params.d_head), 
                    dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    V = torch.randn((params.batch_size, params.seq_len, params.num_heads, params.d_head), 
                    dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    output = torch.empty((params.batch_size, params.seq_len, params.num_heads, params.d_head), 
                        dtype=torch.float32, device="cuda")
    
    cuda_all_inputs = [
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), output.data_ptr(),
        params.batch_size, params.seq_len, params.num_heads, params.d_head
    ]
    return cuda_all_inputs

def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    Q = torch.randn((params.batch_size, params.seq_len, params.num_heads, params.d_head), 
                    dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    K = torch.randn((params.batch_size, params.seq_len, params.num_heads, params.d_head), 
                    dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    V = torch.randn((params.batch_size, params.seq_len, params.num_heads, params.d_head), 
                    dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    output = torch.empty((params.batch_size, params.seq_len, params.num_heads, params.d_head), 
                        dtype=torch.float32, device="cuda")
    
    cuda_output_tensors = [output]
    
    cuda_all_inputs = [
        Q, K, V, output,
        params.batch_size, params.seq_len, params.num_heads, params.d_head
    ]
    torch_all_inputs = [Q, K, V]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def cuda_input_tensor_to_ptr(cuda_all_inputs):
    return [t.data_ptr() if isinstance(t, torch.Tensor) else t for t in cuda_all_inputs]

def cuda_output_tensor_transform(cuda_output):
    return cuda_output  # No transformation needed for MHA
