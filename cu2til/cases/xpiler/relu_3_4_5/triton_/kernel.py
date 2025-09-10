import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the current program id
    pid = tl.program_id(axis=0)
    
    # Compute the block start offset
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for elements within bounds
    mask = offsets < n_elements
    
    # Load data from global memory
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute ReLU: max(0, x)
    relu_output = tl.maximum(0.0, x)
    
    # Store result
    tl.store(output_ptr + offsets, relu_output, mask=mask)

def triton_kernel(x, output, total_elements):
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    # Launch kernel
    relu_kernel[grid](
        x, output, total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
