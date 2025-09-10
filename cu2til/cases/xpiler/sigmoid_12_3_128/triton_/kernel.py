import torch
import triton
import triton.language as tl
import math

@triton.jit
def sigmoid_kernel(
    input_ptr, output_ptr, total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SIGMOID activation function Triton kernel
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = idx < total_elements
    
    # Load input
    x = tl.load(input_ptr + idx, mask=mask)
    
    # Sigmoid computation: 1 / (1 + exp(-x))
    result = tl.sigmoid(x)
    
    tl.store(output_ptr + idx, result, mask=mask)


def triton_kernel(input_tensor):
    """
    Triton SIGMOID implementation
    """
    output_tensor = torch.empty_like(input_tensor)
    total_elements = input_tensor.numel()
    
    # Launch kernel
    grid = (triton.cdiv(total_elements, 1024),)
    
    sigmoid_kernel[grid](
        input_tensor, output_tensor, total_elements,
        BLOCK_SIZE=1024
    )
    
    return output_tensor