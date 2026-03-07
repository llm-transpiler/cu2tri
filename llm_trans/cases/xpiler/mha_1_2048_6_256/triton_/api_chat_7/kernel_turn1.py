import torch
import triton
import triton.language as tl
import math

# The Triton JIT-compiled kernel
@triton.jit
def attention_kernel(
    # Pointers to tensors
    Q, K, V, Out,
    # Stride information for each tensor to handle memory layout
    stride_q_b, stride_q_s, stride_q_h, stride_q_d,
    stride_k_b, stride_k_s, stride_k_h, stride_k_d,
    stride_v_b, stride_v_s, stride_v_h, stride_v_d,
    stride_o_b, stride_o_s, stride_o_h, stride_o_d,
    # Compile-time constants
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Triton kernel for intra-position, inter-head attention.
    Each program instance handles one query from the sequence.
    """
    # 1. Get program IDs to identify the current batch and sequence position
    # This corresponds to `blockIdx.x` and `blockIdx.y` in the CUDA kernel
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    # 2. Create pointers and offsets to load the data for the current query position
    # We will load the entire (NUM_HEADS, HEAD_DIM) block for Q, K, and V.
    
    # Offsets for the head and dimension axes
    offs_h = tl.arange(0, NUM_HEADS)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Create 2D offset matrices for loading/storing
    # Shape: (NUM_HEADS, HEAD_DIM)
    qkv_offs = offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d
    
    # Base pointers for the current (batch, sequence) position
    q_ptr = Q + batch_idx * stride_q_b + seq_idx * stride_q_s
    k_ptr = K + batch_idx * stride_k_b + seq_idx * stride_k_s
    v_ptr = V + batch_idx * stride_v_b + seq_idx * stride_v_s

    # 3. Load Q, K, V from global memory into SRAM (registers)
    # These tensors will have the shape [NUM_HEADS, HEAD_DIM]
    q = tl.load(q_ptr + qkv_offs)
    k = tl.load(k_ptr + qkv_offs)
    v = tl.load(v_ptr + qkv_offs)

    # 4. Compute the first matrix multiplication: S = Q @ K^T
    # q: [NUM_HEADS, HEAD_DIM]
    # tl.trans(k): [HEAD_DIM, NUM_HEADS]
    # s: [NUM_HEADS, NUM_HEADS]
    # This replaces the first nested loop in the CUDA code.
    s = tl.dot(q, tl.trans(k))

    # 5. Apply scaling factor
    # This replaces the `score * scaling_factor` loop
    scaling_factor = (HEAD_DIM ** -0.5).to(s.dtype)
    s = s * scaling_factor

    # 6. Compute softmax over the scores
    # `tl.softmax(s, axis=1)` computes softmax row-wise, which is what the
    # CUDA code does (normalizing over key heads for each query head).
    # This replaces the three separate loops for exp, sum, and division.
    p = tl.softmax(s, axis=1)

    # 7. Compute the final matrix multiplication: Output = P @ V
    # p: [NUM_HEADS, NUM_HEADS]
    # v: [NUM_HEADS, HEAD_DIM]
    # output: [NUM_HEADS, HEAD_DIM]
    # We need to cast `p` to the same dtype as `v` for the dot product.
    p = p.to(v.dtype)
    output = tl.dot(p, v)

    # 8. Write the final result back to global memory
    # Create output pointers and store the [NUM_HEADS, HEAD_DIM] block.
    out_ptr = Out + batch_idx * stride_o_b + seq_idx * stride_o_s
    out_offs = offs_h[:, None] * stride_o_h + offs_d[None, :] * stride_o_d
    tl.store(out_ptr + out_offs, output)


# Python wrapper function to launch the kernel
def triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Args:
        q, k, v: Input tensors of shape [batch_size, seq_len, num_heads, head_dim]
    Returns:
        Output tensor of the same shape.
    """
    # Ensure inputs are on the correct device and have the expected layout
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Create an empty tensor for the output
    output = torch.empty_like(q)
    
    # The grid determines the number of kernel instances to launch.
    # We launch one instance for each query in the batch and sequence.
    grid = (batch_size, seq_len)
    
    # Launch the Triton kernel
    attention_kernel[grid](
        q, k, v, output,
        # Strides for each tensor
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # Constants
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
    )
    
    return output

# --- Verification ---
if __name__ == '__main__':
    # Parameters from the CUDA code
    BATCH_SIZE = 1
    SEQ_LEN = 2048
    NUM_HEADS = 6
    HEAD_DIM = 256

    # Create random input tensors
    torch.manual_seed(0)
    q = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), device='cuda', dtype=torch.float32)
    k = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), device='cuda', dtype=torch.float32)
    v = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), device='cuda', dtype=torch.float32)

    # Run the Triton implementation
    triton_output = triton_attention(q, k, v)

    # --- Naive PyTorch implementation for verification ---
    # This logic directly mirrors the CUDA kernel's loops
    def torch_reference(q, k, v):
        output = torch.zeros_like(q)
        scaling_factor = 1.0 / math.sqrt(HEAD_DIM)
        for i in range(BATCH_SIZE):
            for j in range(SEQ_LEN):
                # Get the data for the current position
                q_ij = q[i, j, :, :] # [num_heads, head_dim]
                k_ij = k[i, j, :, :] # [num_heads, head_dim]
                v_ij = v[i, j, :, :] # [num_heads, head_dim]
                
                # S = Q @ K.T
                score = torch.matmul(q_ij, k_ij.transpose(-1, -2)) # [num_heads, num_heads]
                
                # Scale and softmax
                score = score * scaling_factor
                p = torch.nn.functional.softmax(score, dim=-1)
                
                # O = P @ V
                out_ij = torch.matmul(p, v_ij)
                output[i, j, :, :] = out_ij
        return output

    torch_output = torch_reference(q, k, v)

    # Compare results
    print("Triton and PyTorch outputs are close:", torch.allclose(triton_output, torch_output, atol=1e-4, rtol=1e-4))
    
    # %timeit triton_attention(q, k, v)
    # %timeit torch_reference(q, k, v)
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/xpiler/mha_1_2048_6_256/triton_/api_chat_7/kernel_turn1.py
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 42, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 1654, in arange
    return _semantic.arange(start, end)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/semantic.py", line 583, in arange
    raise ValueError("arange's range must be a power of 2")
ValueError: arange's range must be a power of 2

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/workspace/cu2til/cases/xpiler/mha_1_2048_6_256/triton_/api_chat_7/kernel_turn1.py", line 135, in <module>
    triton_output = triton_attention(q, k, v)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/cu2til/cases/xpiler/mha_1_2048_6_256/triton_/api_chat_7/kernel_turn1.py", line 106, in triton_attention
    attention_kernel[grid](
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
triton.compiler.errors.CompilationError: at 26:13:
    Each program instance handles one query from the sequence.
    """
    # 1. Get program IDs to identify the current batch and sequence position
    # This corresponds to `blockIdx.x` and `blockIdx.y` in the CUDA kernel
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    # 2. Create pointers and offsets to load the data for the current query position
    # We will load the entire (NUM_HEADS, HEAD_DIM) block for Q, K, and V.

    # Offsets for the head and dimension axes
    offs_h = tl.arange(0, NUM_HEADS)
             ^
'''