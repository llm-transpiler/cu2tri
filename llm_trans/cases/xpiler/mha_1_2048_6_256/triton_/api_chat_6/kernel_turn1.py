import torch
import triton
import triton.language as tl

@triton.jit
def attention_kernel_triton(
    # Pointers to tensors
    Q_ptr, K_ptr, V_ptr, output_ptr,
    # Stride variables for memory access
    stride_q_bs, stride_q_sl, stride_q_h, stride_q_d,
    stride_k_bs, stride_k_sl, stride_k_h, stride_k_d,
    stride_v_bs, stride_v_sl, stride_v_h, stride_v_d,
    stride_o_bs, stride_o_sl, stride_o_h, stride_o_d,
    # Matrix dimensions
    num_heads,
    head_dim,
    # Triton block sizes (powers of 2 for performance)
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Triton kernel for intra-token head attention.
    Each program instance computes the attention for a single token across all its heads.
    """
    # 1. Get program IDs to identify the current token
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    # 2. Compute pointers to the start of the Q, K, V slices for the current token
    q_start_ptr = Q_ptr + batch_idx * stride_q_bs + seq_idx * stride_q_sl
    k_start_ptr = K_ptr + batch_idx * stride_k_bs + seq_idx * stride_k_sl
    v_start_ptr = V_ptr + batch_idx * stride_v_bs + seq_idx * stride_v_sl

    # 3. Create ranges and masks for loading data blocks
    # Ranges for the head and dimension axes
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Create 2D blocks of pointers for Q and V
    q_ptrs = q_start_ptr + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d
    v_ptrs = v_start_ptr + offs_h[:, None] * stride_v_h + offs_d[None, :] * stride_v_d
    
    # Create a transposed 2D block of pointers for K to load K.T directly
    k_t_ptrs = k_start_ptr + offs_d[:, None] * stride_k_d + offs_h[None, :] * stride_k_h

    # Create masks to prevent out-of-bounds access if dimensions are not powers of 2
    mask_h = offs_h < num_heads
    mask_d = offs_d < head_dim # Not strictly needed if BLOCK_D == head_dim

    # 4. Load Q, K.T, and V slices from HBM into SRAM
    # Load Q slice [num_heads, head_dim]
    q = tl.load(q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
    # Load K.T slice [head_dim, num_heads]
    k_t = tl.load(k_t_ptrs, mask=mask_h[None, :] & mask_d[:, None], other=0.0)
    # Load V slice [num_heads, head_dim]
    v = tl.load(v_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)

    # 5. Compute S = Q @ K.T
    # The result 'scores' will have shape [BLOCK_H, BLOCK_H]
    scores = tl.dot(q, k_t)

    # 6. Apply scaling and softmax
    scaling_factor = (1.0 / tl.sqrt(head_dim.to(tl.float32)))
    scores *= scaling_factor

    # Create a mask for the score matrix to handle padding in the head dimension
    score_mask = mask_h[:, None] & mask_h[None, :]
    # Set scores for padded heads to -inf before softmax
    scores = tl.where(score_mask, scores, -float('inf'))

    # Compute softmax row-wise
    row_max = tl.max(scores, axis=1)
    scores -= row_max[:, None]  # For numerical stability
    numerator = tl.exp(scores)
    denominator = tl.sum(numerator, axis=1)
    softmax_scores = numerator / denominator[:, None]
    
    # Zero out any NaNs that might have resulted from empty rows
    softmax_scores = tl.where(score_mask, softmax_scores, 0.0)

    # 7. Compute final output O = P @ V
    # P is [num_heads, num_heads], V is [num_heads, head_dim]
    # The result 'output' will have shape [BLOCK_H, BLOCK_D]
    # Ensure dtypes match for tl.dot
    output = tl.dot(softmax_scores.to(v.dtype), v)

    # 8. Write the output block back to HBM
    output_start_ptr = output_ptr + batch_idx * stride_o_bs + seq_idx * stride_o_sl
    output_ptrs = output_start_ptr + offs_h[:, None] * stride_o_h + offs_d[None, :] * stride_o_d
    output_mask = mask_h[:, None] & mask_d[None, :]
    tl.store(output_ptrs, output, mask=output_mask)


def triton_kernel_launcher(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    Python launcher for the Triton attention kernel.
    """
    # Ensure inputs are on the correct device and have the expected layout
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    
    # Get tensor dimensions
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Create an empty output tensor
    output = torch.empty_like(q)
    
    # Define the grid for the kernel launch
    # One program per token in the batch
    grid = (batch_size, seq_len)

    # Define Triton block sizes. They should be powers of 2 for best performance.
    BLOCK_H = triton.next_power_of_2(num_heads)
    BLOCK_D = triton.next_power_of_2(head_dim)

    # Launch the kernel
    attention_kernel_triton[grid](
        # Tensors
        q, k, v, output,
        # Strides for Q
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # Strides for K
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # Strides for V
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # Strides for Output
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # Dimensions
        num_heads,
        head_dim,
        # Block sizes
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
    )
    
    return output

# --- Example Usage and Verification ---
if __name__ == '__main__':
    # Parameters from the CUDA kernel
    batch_size = 1
    seq_len = 2048
    num_heads = 6
    head_dim = 256

    # Create random input tensors
    torch.manual_seed(0)
    q = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda', dtype=torch.float32)
    k = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda', dtype=torch.float32)
    v = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda', dtype=torch.float32)

    # Run the Triton kernel
    triton_output = triton_kernel_launcher(q, k, v)

    # --- PyTorch reference implementation for verification ---
    # This performs the same logic as the CUDA/Triton kernel
    import math
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    p = torch.nn.functional.softmax(scores, dim=-1)
    pytorch_output = torch.matmul(p, v)
    
    # Compare results
    print("Triton vs. PyTorch allclose:", torch.allclose(triton_output, pytorch_output, atol=1e-5, rtol=1e-4))
    
    # Print a small slice to visually inspect
    print("\nTriton output (slice):")
    print(triton_output[0, 0, :3, :5])
    
    print("\nPyTorch output (slice):")
    print(pytorch_output[0, 0, :3, :5])
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/xpiler/mha_1_2048_6_256/triton_/api_chat_6/kernel_turn1.py
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 42, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 2045, in dot
    return _semantic.dot(input, other, acc, input_precision, max_num_imprecise_acc, out_dtype)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/semantic.py", line 1503, in dot
    assert lhs.shape[-2].value >= min_dot_size[0] and lhs.shape[-1].value >= min_dot_size[2] \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Input shapes should have M >= 16, N >= 16 and K >= 16

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/workspace/cu2til/cases/xpiler/mha_1_2048_6_256/triton_/api_chat_6/kernel_turn1.py", line 153, in <module>
    triton_output = triton_kernel_launcher(q, k, v)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/cu2til/cases/xpiler/mha_1_2048_6_256/triton_/api_chat_6/kernel_turn1.py", line 117, in triton_kernel_launcher
    attention_kernel_triton[grid](
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
triton.compiler.errors.CompilationError: at 55:13:

    # 4. Load Q, K.T, and V slices from HBM into SRAM
    # Load Q slice [num_heads, head_dim]
    q = tl.load(q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
    # Load K.T slice [head_dim, num_heads]
    k_t = tl.load(k_t_ptrs, mask=mask_h[None, :] & mask_d[:, None], other=0.0)
    # Load V slice [num_heads, head_dim]
    v = tl.load(v_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)

    # 5. Compute S = Q @ K.T
    # The result 'scores' will have shape [BLOCK_H, BLOCK_H]
    scores = tl.dot(q, k_t)
             ^
'''