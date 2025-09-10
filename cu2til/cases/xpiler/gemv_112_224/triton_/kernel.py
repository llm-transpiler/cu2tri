import torch
import triton
import triton.language as tl

@triton.jit
def gemv_kernel(a_ptr, x_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    # Get the current program id - each handles one row of the output vector
    pid = tl.program_id(axis=0)
    
    # Each program handles one row of the matrix
    if pid >= M:
        return
    
    # Compute dot product for row pid
    row_sum = 0.0
    
    # Process N elements in blocks
    for block_start in range(0, N, BLOCK_SIZE):
        # Calculate offsets for this block
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        # Load A[pid, offsets] and x[offsets]
        a_ptrs = a_ptr + pid * N + offsets
        a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
        
        x_ptrs = x_ptr + offsets
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Compute partial dot product
        row_sum += tl.sum(a_vals * x_vals)
    
    # Store result
    tl.store(y_ptr + pid, row_sum)

def triton_kernel(A, x, y, m, n):
    # Choose block size for vector elements
    BLOCK_SIZE = 64
    
    # Launch one program per output row
    grid = (m,)
    
    # Launch kernel
    gemv_kernel[grid](
        A, x, y, m, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
