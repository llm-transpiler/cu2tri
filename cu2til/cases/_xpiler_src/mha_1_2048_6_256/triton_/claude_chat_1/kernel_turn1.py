import triton
import triton.language as tl
import torch


@triton.jit
def mha_kernel(
    Q, K, V, Output,
    batch_size, seq_len, num_heads, head_dim,
    stride_q_batch, stride_q_seq, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_seq, stride_k_head, stride_k_dim,
    stride_v_batch, stride_v_seq, stride_v_head, stride_v_dim,
    stride_o_batch, stride_o_seq, stride_o_head, stride_o_dim,
    BLOCK_SIZE_HEAD: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    # Scaling factor
    scaling_factor = 1.0 / tl.sqrt(head_dim.to(tl.float32))
    
    # Calculate base offsets for current batch and sequence position
    q_offset = (batch_idx * stride_q_batch + 
                seq_idx * stride_q_seq + 
                head_idx * stride_q_head)
    
    k_offset_base = (batch_idx * stride_k_batch + 
                     seq_idx * stride_k_seq)
    
    v_offset_base = (batch_idx * stride_v_batch + 
                     seq_idx * stride_v_seq)
    
    output_offset = (batch_idx * stride_o_batch + 
                     seq_idx * stride_o_seq + 
                     head_idx * stride_o_head)
    
    # Load Q vector for current head
    dim_range = tl.arange(0, BLOCK_SIZE_DIM)
    dim_mask = dim_range < head_dim
    q_ptrs = Q + q_offset + dim_range
    q_vec = tl.load(q_ptrs, mask=dim_mask, other=0.0)
    
    # Compute attention scores for all heads
    scores = tl.zeros([BLOCK_SIZE_HEAD], dtype=tl.float32)
    
    for head_k in range(num_heads):
        k_offset = k_offset_base + head_k * stride_k_head
        k_ptrs = K + k_offset + dim_range
        k_vec = tl.load(k_ptrs, mask=dim_mask, other=0.0)
        
        # Compute dot product
        score = tl.sum(q_vec * k_vec)
        
        # Store score (only if head_k < BLOCK_SIZE_HEAD for safety)
        if head_k < BLOCK_SIZE_HEAD:
            scores = tl.where(tl.arange(0, BLOCK_SIZE_HEAD) == head_k, 
                             score, scores)
    
    # Apply scaling
    scores = scores * scaling_factor
    
    # Apply softmax
    scores_max = tl.max(scores, axis=0)
    scores_exp = tl.exp(scores - scores_max)
    scores_sum = tl.sum(scores_exp, axis=0)
    attention_weights = scores_exp / scores_sum
    
    # Compute final output
    output_vec = tl.zeros([BLOCK_SIZE_DIM], dtype=tl.float32)
    
    for head_v in range(num_heads):
        v_offset = v_offset_base + head_v * stride_v_head
        v_ptrs = V + v_offset + dim_range
        v_vec = tl.load(v_ptrs, mask=dim_mask, other=0.0)
        
        # Get attention weight for this head
        weight = tl.load(attention_weights + head_v) if head_v < BLOCK_SIZE_HEAD else 0.0
        
        # Accumulate weighted values
        output_vec += weight * v_vec
    
    # Store output
    output_ptrs = Output + output_offset + dim_range
    tl.store(output_ptrs, output_vec, mask=dim_mask)


def mha_triton(q, k, v):
    """
    Multi-head attention using Triton
    
    Args:
        q, k, v: [batch_size, seq_len, num_heads, head_dim]
    
    Returns:
        output: [batch_size, seq_len, num_heads, head_dim]
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Allocate output
    output = torch.empty_like(q)
    
    # Define block sizes
    BLOCK_SIZE_HEAD = triton.next_power_of_2(num_heads)
    BLOCK_SIZE_DIM = triton.next_power_of_2(head_dim)
    
    # Launch kernel
    grid = (batch_size, seq_len, 1)
    
    mha_kernel[grid](
        q, k, v, output,
        batch_size, seq_len, num_heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3), 
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE_HEAD=BLOCK_SIZE_HEAD,
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
    )
    
    return output


# Example usage
if __name__ == "__main__":
    batch_size, seq_len, num_heads, head_dim = 1, 2048, 6, 256
    device = "cuda"
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device) 
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    
    output = mha_triton(q, k, v)
    print(f"Output shape: {output.shape}")
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/mha_1_2048_6_256/triton/kernel.py
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 42, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 2150, in load
    return _semantic.load(pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/semantic.py", line 1086, in load
    return self._load_legacy(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/semantic.py", line 1017, in _load_legacy
    raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")
ValueError: Unsupported ptr type <['8'], fp32> in `tl.load`

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/workspace/cu2til/cases/mha_1_2048_6_256/triton/kernel.py", line 135, in <module>
    output = mha_triton(q, k, v)
             ^^^^^^^^^^^^^^^^^^^
  File "/workspace/cu2til/cases/mha_1_2048_6_256/triton/kernel.py", line 112, in mha_triton
    mha_kernel[grid](
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 390, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 594, in run
    kernel = self.compile(src, target=target, options=options.__dict__)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py", line 339, in compile
    module = src.make_ir(options, codegen_fns, module_map, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py", line 83, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 74:17:
    attention_weights = scores_exp / scores_sum

    # Compute final output
    output_vec = tl.zeros([BLOCK_SIZE_DIM], dtype=tl.float32)

    for head_v in range(num_heads):
        v_offset = v_offset_base + head_v * stride_v_head
        v_ptrs = V + v_offset + dim_range
        v_vec = tl.load(v_ptrs, mask=dim_mask, other=0.0)

        # Get attention weight for this head
        weight = tl.load(attention_weights + head_v) if head_v < BLOCK_SIZE_HEAD else 0.0
                 ^
'''