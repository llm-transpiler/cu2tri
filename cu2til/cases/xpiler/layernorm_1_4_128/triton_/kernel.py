import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel(x_ptr, gamma_ptr, beta_ptr, output_ptr, batch_size, seq_length, d_model, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get the current program id - each handles one sequence position
    pid = tl.program_id(axis=0)
    
    # Calculate total number of positions
    total_positions = batch_size * seq_length
    
    if pid >= total_positions:
        return
    
    # Calculate the starting offset for this position
    pos_offset = pid * d_model
    
    # Load the input vector for this position
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < d_model
    
    x_ptrs = x_ptr + pos_offset + offsets
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x_vals) / d_model
    
    # Compute variance
    x_centered = x_vals - mean
    variance = tl.sum(x_centered * x_centered) / d_model
    
    # Compute standard deviation with epsilon
    std = tl.sqrt(variance + eps)
    
    # Load gamma and beta
    gamma_ptrs = gamma_ptr + offsets
    beta_ptrs = beta_ptr + offsets
    gamma_vals = tl.load(gamma_ptrs, mask=mask, other=1.0)
    beta_vals = tl.load(beta_ptrs, mask=mask, other=0.0)
    
    # Normalize and scale
    normalized = (x_vals - mean) / std
    output_vals = gamma_vals * normalized + beta_vals
    
    # Store result
    output_ptrs = output_ptr + pos_offset + offsets
    tl.store(output_ptrs, output_vals, mask=mask)

def triton_kernel(x, gamma, beta, output, batch_size, seq_length, d_model):
    # LayerNorm epsilon
    eps = 1e-5
    
    # Block size should cover the feature dimension
    BLOCK_SIZE = triton.next_power_of_2(d_model)
    
    # Calculate total number of positions to normalize
    total_positions = batch_size * seq_length
    
    # Launch one program per position
    grid = (total_positions,)
    
    # Launch kernel
    layernorm_kernel[grid](
        x, gamma, beta, output, batch_size, seq_length, d_model, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
