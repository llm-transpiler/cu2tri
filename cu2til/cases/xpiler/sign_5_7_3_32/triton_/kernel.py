import torch
import triton
import triton.language as tl
import math

@triton.jit
def sign_kernel(
    input_ptr, output_ptr, total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Sign activation function Triton kernel
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = idx < total_elements
    
    # Load input
    x = tl.load(input_ptr + idx, mask=mask)
    
    # Sign computation: sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0
    result = tl.where(x > 0.0, 1.0, tl.where(x < 0.0, -1.0, 0.0))
    
    tl.store(output_ptr + idx, result, mask=mask)


def triton_kernel(input_tensor):
    """
    Triton Sign implementation
    """
    output_tensor = torch.empty_like(input_tensor)
    total_elements = input_tensor.numel()
    
    # Launch kernel
    grid = (triton.cdiv(total_elements, 1024),)
    
    sign_kernel[grid](
        input_tensor, output_tensor, total_elements,
        BLOCK_SIZE=1024
    )
    
    return output_tensor