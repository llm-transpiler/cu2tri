import torch
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    input_ptr, kernel_ptr, output_ptr,
    input_size, kernel_size, output_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    1D Convolution Triton kernel
    """
    idx = tl.program_id(axis=0)
    
    if idx >= output_size:
        return
    
    # Compute convolution at position idx
    result = 0.0
    for k in range(kernel_size):
        input_idx = idx + k
        if input_idx < input_size:
            input_val = tl.load(input_ptr + input_idx)
            kernel_val = tl.load(kernel_ptr + k)
            result += input_val * kernel_val
    
    tl.store(output_ptr + idx, result)


def triton_kernel(input_tensor, kernel_tensor):
    """
    Triton conv1d implementation
    """
    input_size = input_tensor.shape[0]
    kernel_size = kernel_tensor.shape[0]
    output_size = input_size - kernel_size + 1
    
    output_tensor = torch.empty((output_size,), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    grid = (triton.cdiv(output_size, 1024),)
    
    conv1d_kernel[grid](
        input_tensor, kernel_tensor, output_tensor,
        input_size, kernel_size, output_size,
        BLOCK_SIZE=1024
    )
    
    return output_tensor