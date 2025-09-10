import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, output_ptr, seq_length, d_model, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get the current program id - each handles one sequence position
    pid = tl.program_id(axis=0)
    
    if pid >= seq_length:
        return
    
    # Calculate the starting offset for this position
    pos_offset = pid * d_model
    
    # Load the input vector for this position
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < d_model
    
    x_ptrs = x_ptr + pos_offset + offsets
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Compute mean squared (RMS)
    x_squared = x_vals * x_vals
    mean_squared = tl.sum(x_squared) / d_model
    
    # Compute RMS normalization: x / sqrt(mean_squared + eps)
    rms = tl.sqrt(mean_squared + eps)
    normalized = x_vals / rms
    
    # Store result
    output_ptrs = output_ptr + pos_offset + offsets
    tl.store(output_ptrs, normalized, mask=mask)

def triton_kernel(x, output, seq_length, d_model):
    # RMSNorm epsilon
    eps = 1e-6
    
    # Block size should cover the feature dimension
    BLOCK_SIZE = triton.next_power_of_2(d_model)
    
    # Launch one program per sequence position
    grid = (seq_length,)
    
    # Launch kernel
    rmsnorm_kernel[grid](
        x, output, seq_length, d_model, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
