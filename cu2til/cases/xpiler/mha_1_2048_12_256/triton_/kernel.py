import torch
import triton
import triton.language as tl

@triton.jit
def mha_kernel(q_ptr, k_ptr, v_ptr, output_ptr, 
               batch_size, seq_len, num_heads, head_dim, scale_factor: tl.constexpr,
               BLOCK_SIZE_SEQ: tl.constexpr, BLOCK_SIZE_HEAD: tl.constexpr):
    # Get program IDs
    pid_batch = tl.program_id(axis=0)
    pid_seq = tl.program_id(axis=1)
    
    if pid_batch >= batch_size or pid_seq >= seq_len:
        return
    
    # Calculate base offsets for this (batch, seq) position
    base_offset = pid_batch * seq_len * num_heads * head_dim + pid_seq * num_heads * head_dim
    
    # For each head
    for head_idx in range(num_heads):
        head_offset = base_offset + head_idx * head_dim
        
        # Load Q vector for this head (1, head_dim)
        head_offsets = tl.arange(0, BLOCK_SIZE_HEAD)
        head_mask = head_offsets < head_dim
        
        q_ptrs = q_ptr + head_offset + head_offsets
        q_vals = tl.load(q_ptrs, mask=head_mask, other=0.0)
        
        # Initialize attention output
        attention_output = tl.zeros((BLOCK_SIZE_HEAD,), dtype=tl.float32)
        
        # Compute attention for this position against all positions
        attention_sum = 0.0
        
        # First pass: compute attention weights and find max for stability
        max_score = -float('inf')
        for seq_j in range(seq_len):
            k_offset = pid_batch * seq_len * num_heads * head_dim + seq_j * num_heads * head_dim + head_idx * head_dim
            k_ptrs = k_ptr + k_offset + head_offsets
            k_vals = tl.load(k_ptrs, mask=head_mask, other=0.0)
            
            # Compute attention score: Q @ K.T / sqrt(head_dim)
            score = tl.sum(q_vals * k_vals) / scale_factor
            max_score = tl.maximum(max_score, score)
        
        # Second pass: compute softmax and weighted values
        for seq_j in range(seq_len):
            k_offset = pid_batch * seq_len * num_heads * head_dim + seq_j * num_heads * head_dim + head_idx * head_dim
            v_offset = k_offset  # V has same layout as K
            
            k_ptrs = k_ptr + k_offset + head_offsets
            v_ptrs = v_ptr + v_offset + head_offsets
            
            k_vals = tl.load(k_ptrs, mask=head_mask, other=0.0)
            v_vals = tl.load(v_ptrs, mask=head_mask, other=0.0)
            
            # Compute attention score and apply softmax
            score = tl.sum(q_vals * k_vals) / scale_factor
            exp_score = tl.exp(score - max_score)
            attention_sum += exp_score
            
            # Accumulate weighted values
            attention_output += exp_score * v_vals
        
        # Normalize by attention sum (complete softmax)
        attention_output = attention_output / attention_sum
        
        # Store result for this head
        output_ptrs = output_ptr + head_offset + head_offsets
        tl.store(output_ptrs, attention_output, mask=head_mask)

def triton_kernel(q, k, v, output, batch_size, seq_len, num_heads, head_dim):
    scale_factor = head_dim ** 0.5
    
    # Block sizes
    BLOCK_SIZE_SEQ = 1  # Process one sequence position at a time
    BLOCK_SIZE_HEAD = triton.next_power_of_2(head_dim)
    
    # Grid dimensions
    grid = (batch_size, seq_len)
    
    # Launch kernel
    mha_kernel[grid](
        q, k, v, output,
        batch_size, seq_len, num_heads, head_dim, scale_factor,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
        BLOCK_SIZE_HEAD=BLOCK_SIZE_HEAD
    )
