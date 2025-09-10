import torch
import ctypes
from dataclasses import dataclass
from cu2til.tools.builder import SEED
from cu2til.tools.layout import convert_nhwc_to_nchw

@dataclass
class Params:
    """MAXPOOL operation parameters"""
    batch_size: int = 5
    height: int = 32
    width: int = 32
    channels: int = 64
    kernel_size: int = 5
    stride: int = 3
    
    output_height: int = 10
    output_width: int = 10

def get_cuda_argtypes():
    return [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]

def get_cuda_inputs(params: Params):
    torch.manual_seed(SEED)
    input_tensor_nhwc = torch.randn(
        (params.batch_size, params.height, params.width, params.channels), 
        dtype=torch.float32, device="cuda"
    ).normal_(mean=0.0, std=0.5)
    
    output_tensor_nhwc = torch.empty(
        (params.batch_size, params.output_height, params.output_width, params.channels),
        dtype=torch.float32, device="cuda"
    )
    
    cuda_all_inputs = [
        input_tensor_nhwc.data_ptr(), output_tensor_nhwc.data_ptr(),
        params.batch_size, params.channels, params.height,
        params.kernel_size, params.stride
    ]
    return cuda_all_inputs

def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    input_tensor_nhwc = torch.randn(
        (params.batch_size, params.height, params.width, params.channels), 
        dtype=torch.float32, device="cuda"
    ).normal_(mean=0.0, std=0.5)
    
    output_tensor_nhwc = torch.empty(
        (params.batch_size, params.output_height, params.output_width, params.channels),
        dtype=torch.float32, device="cuda"
    )
    
    cuda_output_tensors = [output_tensor_nhwc]
    input_tensor_nchw = convert_nhwc_to_nchw(input_tensor_nhwc)
    
    cuda_all_inputs = [
        input_tensor_nhwc, output_tensor_nhwc,
        params.batch_size, params.channels, params.height,
        params.kernel_size, params.stride
    ]
    torch_all_inputs = [input_tensor_nchw, params.kernel_size, params.stride]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def cuda_input_tensor_to_ptr(cuda_all_inputs):
    return [t.data_ptr() if isinstance(t, torch.Tensor) else t for t in cuda_all_inputs]

def cuda_output_tensor_transform(cuda_output_nhwc):
    return convert_nhwc_to_nchw(cuda_output_nhwc)
