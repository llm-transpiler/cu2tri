import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the current program id
    pid = tl.program_id(axis=0)
    
    # Compute the block start offset
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for elements within bounds
    mask = offsets < n_elements
    
    # Load data from global memory
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute Sigmoid: 1 / (1 + exp(-x))
    neg_x = -x
    exp_neg_x = tl.exp(neg_x)
    one_plus_exp_neg_x = 1.0 + exp_neg_x
    sigmoid_output = 1.0 / one_plus_exp_neg_x
    
    # Store result
    tl.store(output_ptr + offsets, sigmoid_output, mask=mask)

def triton_kernel(x, output, total_elements):
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    # Launch kernel
    sigmoid_kernel[grid](
        x, output, total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
