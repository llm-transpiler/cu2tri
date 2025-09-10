import torch
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel(
    input_ptr, output_ptr, total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    GELU activation function Triton kernel
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = idx < total_elements
    
    # Load input
    x = tl.load(input_ptr + idx, mask=mask)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608  # sqrt(2/π)
    coeff = 0.044715
    
    # Compute x^3
    x_cubed = x * x * x
    
    # Compute the argument to tanh
    tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed)
    
    # Compute tanh using the identity: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    # But we'll use a simpler approximation for numerical stability
    tanh_val = tl.tanh(tanh_arg)
    
    # Final GELU computation
    result = 0.5 * x * (1.0 + tanh_val)
    
    tl.store(output_ptr + idx, result, mask=mask)


def triton_kernel(input_tensor):
    """
    Triton GELU implementation
    """
    output_tensor = torch.empty_like(input_tensor)
    total_elements = input_tensor.numel()
    
    # Launch kernel
    grid = (triton.cdiv(total_elements, 1024),)
    
    gelu_kernel[grid](
        input_tensor, output_tensor, total_elements,
        BLOCK_SIZE=1024
    )
    
    return output_tensor