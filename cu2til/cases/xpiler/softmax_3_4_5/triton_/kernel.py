import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, output_ptr, batch_size, seq_len, feature_dim, BLOCK_SIZE: tl.constexpr):
    # Get the current program id - each handles one sequence in the batch
    pid = tl.program_id(axis=0)
    
    # Calculate total number of sequences
    total_sequences = batch_size * seq_len
    
    if pid >= total_sequences:
        return
    
    # Calculate the starting offset for this sequence
    seq_offset = pid * feature_dim
    
    # Load the input sequence
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < feature_dim
    
    x_ptrs = x_ptr + seq_offset + offsets
    x_vals = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability (subtract max before exp)
    max_val = tl.max(x_vals, axis=0)
    
    # Compute exp(x - max)
    exp_vals = tl.exp(x_vals - max_val)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_vals, axis=0)
    
    # Compute softmax = exp(x - max) / sum_exp
    softmax_vals = exp_vals / sum_exp
    
    # Store result
    output_ptrs = output_ptr + seq_offset + offsets
    tl.store(output_ptrs, softmax_vals, mask=mask)

def triton_kernel(x, output, batch_size, seq_len, feature_dim):
    # Block size should cover the feature dimension
    BLOCK_SIZE = triton.next_power_of_2(feature_dim)
    
    # Calculate total number of sequences to process
    total_sequences = batch_size * seq_len
    
    # Launch one program per sequence
    grid = (total_sequences,)
    
    # Launch kernel
    softmax_kernel[grid](
        x, output, batch_size, seq_len, feature_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
