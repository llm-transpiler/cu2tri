import torch
import ctypes
from dataclasses import dataclass
from llm_trans.tools.checker import SEED

@dataclass
class Params:
    input_shape: tuple = (128, 256)
    output_shape: tuple = (256,)
    rows: int = 128
    inner: int = 256

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # output
        ctypes.c_int,     # rows
        ctypes.c_int      # inner
    ]

def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    x = torch.randn(params.input_shape, dtype=torch.float32, device="cuda")
    output = torch.empty(params.output_shape, dtype=torch.float32, device="cuda")
    cuda_all_inputs = [x, output, params.rows, params.inner]
    torch_all_inputs = [x]
    return cuda_all_inputs, torch_all_inputs, [output]

def cuda_output_tensor_transform(tensor):
    return tensor
