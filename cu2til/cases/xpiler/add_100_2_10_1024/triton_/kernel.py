import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the current program id
    pid = tl.program_id(axis=0)
    
    # Compute the block start offset
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for elements within bounds
    mask = offsets < n_elements
    
    # Load data from global memory
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute the addition
    output = a + b
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_kernel(A, B, output, total_elements):
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    # Launch kernel
    add_kernel[grid](
        A, B, output, total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
