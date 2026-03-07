import torch
import triton
import triton.language as tl
import math

# Triton Kernel
@triton.jit
def mha_triton_kernel(
    # Pointers to tensors
    Q_ptr, K_ptr, V_ptr,
    output_ptr,
    # Stride information for correct memory access
    stride_q_b, stride_q_s, stride_q_h,
    stride_k_b, stride_k_s, stride_k_h,
    stride_v_b, stride_v_s, stride_v_h,
    stride_o_b, stride_o_s, stride_o_h,
    # Other parameters
    SEQ_LEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    """
    Triton kernel for a non-standard "cross-head" attention mechanism.
    Each instance of this kernel computes the attention for one item in the batch
    and one position in the sequence.

    Grid: (BATCH_SIZE, SEQ_LEN)
    """
    # 1. Get program IDs to identify the current (batch, seq_pos)
    pid_batch = tl.program_id(axis=0)
    pid_seq = tl.program_id(axis=1)

    # 2. Load Q, K, V blocks for the current (batch, seq_pos)
    # We need to load all heads for this position. Shape: [NUM_HEADS, HEAD_DIM]
    
    # Offsets for the head and dimension axes
    offs_h = tl.arange(0, NUM_HEADS)
    offs_d = tl.arange(0, HEAD_DIM)

    # Create pointers for the Q, K, V blocks
    # Pointer to the start of the current (batch, seq_pos) slice
    q_start_ptr = Q_ptr + pid_batch * stride_q_b + pid_seq * stride_q_s
    k_start_ptr = K_ptr + pid_batch * stride_k_b + pid_seq * stride_k_s
    v_start_ptr = V_ptr + pid_batch * stride_v_b + pid_seq * stride_v_s

    # Create 2D blocks of pointers
    # offs_h[:, None] -> shape [NUM_HEADS, 1]
    # offs_d[None, :] -> shape [1, HEAD_DIM]
    # Broadcasting creates a [NUM_HEADS, HEAD_DIM] matrix of offsets
    q_ptrs = q_start_ptr + offs_h[:, None] * stride_q_h + offs_d[None, :]
    k_ptrs = k_start_ptr + offs_h[:, None] * stride_k_h + offs_d[None, :]
    v_ptrs = v_start_ptr + offs_h[:, None] * stride_v_h + offs_d[None, :]

    # Load the data into SRAM
    # q, k, v will have shape [NUM_HEADS, HEAD_DIM]
    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)

    # 3. Compute Q @ K.T
    # q: [NUM_HEADS, HEAD_DIM], k: [NUM_HEADS, HEAD_DIM]
    # We need to compute dot(q, k.T)
    # tl.dot handles this automatically if we transpose k
    scores = tl.dot(q, tl.trans(k)) # Result shape: [NUM_HEADS, NUM_HEADS]

    # 4. Apply scaling factor
    scaling_factor = 1.0 / math.sqrt(HEAD_DIM)
    scores = scores * scaling_factor

    # 5. Apply softmax
    # tl.softmax performs row-wise softmax by default on the last axis
    attn_weights = tl.softmax(scores, axis=1)
    # Ensure the computation happens in float32 for precision
    attn_weights = attn_weights.to(tl.float32)

    # 6. Compute Softmax @ V
    # attn_weights: [NUM_HEADS, NUM_HEADS], v: [NUM_HEADS, HEAD_DIM]
    # The result will have shape [NUM_HEADS, HEAD_DIM]
    output = tl.dot(attn_weights, v)

    # 7. Store the result back to global memory
    # Create output pointers similar to input pointers
    output_start_ptr = output_ptr + pid_batch * stride_o_b + pid_seq * stride_o_s
    output_ptrs = output_start_ptr + offs_h[:, None] * stride_o_h + offs_d[None, :]
    tl.store(output_ptrs, output)


# Wrapper function to launch the Triton kernel
def mha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper for the cross-head attention mechanism.

    Args:
        q (torch.Tensor): Query tensor of shape (B, S, H, D)
        k (torch.Tensor): Key tensor of shape (B, S, H, D)
        v (torch.Tensor): Value tensor of shape (B, S, H, D)

    Returns:
        torch.Tensor: Output tensor of shape (B, S, H, D)
    """
    # Get tensor dimensions
    BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM = q.shape
    
    # Create an empty output tensor
    output = torch.empty_like(q)

    # Ensure tensors are contiguous in memory for simplicity,
    # although strides handle non-contiguous layouts.
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    # Define the grid for launching the kernel
    # One kernel instance per (batch, seq_pos)
    grid = (BATCH, SEQ_LEN)

    # Launch the kernel
    mha_triton_kernel[grid](
        # Tensors
        q, k, v, output,
        # Strides for Q
        q.stride(0), q.stride(1), q.stride(2),
        # Strides for K
        k.stride(0), k.stride(1), k.stride(2),
        # Strides for V
        v.stride(0), v.stride(1), v.stride(2),
        # Strides for Output
        output.stride(0), output.stride(1), output.stride(2),
        # Other parameters (as constexpr)
        SEQ_LEN=SEQ_LEN,
        NUM_HEADS=NUM_HEADS,
        HEAD_DIM=HEAD_DIM,
    )
    return output

# --- Verification ---
def torch_reference_mha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch implementation for verification."""
    BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM = q.shape
    
    # This logic is unusual, so we implement it carefully in a loop
    # to match the kernel's behavior exactly.
    output = torch.zeros_like(q)
    scaling_factor = 1.0 / math.sqrt(HEAD_DIM)
    
    for b in range(BATCH):
        for s in range(SEQ_LEN):
            # Get slices for the current (batch, seq_pos)
            q_slice = q[b, s, :, :]  # Shape: [NUM_HEADS, HEAD_DIM]
            k_slice = k[b, s, :, :]  # Shape: [NUM_HEADS, HEAD_DIM]
            v_slice = v[b, s, :, :]  # Shape: [NUM_HEADS, HEAD_DIM]
            
            # 1. Q @ K.T
            scores = torch.matmul(q_slice, k_slice.T) # Shape: [NUM_HEADS, NUM_HEADS]
            
            # 2. Scale
            scores *= scaling_factor
            
            # 3. Softmax
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            
            # 4. Softmax @ V
            output_slice = torch.matmul(attn_weights, v_slice) # Shape: [NUM_HEADS, HEAD_DIM]
            
            # Store result
            output[b, s, :, :] = output_slice
            
    return output


if __name__ == "__main__":
    # Parameters from the CUDA code
    BATCH_SIZE = 4
    SEQ_LEN = 2048
    NUM_HEADS = 6
    HEAD_DIM = 256

    # Create random input tensors
    # Note: The CUDA code uses a flattened layout. Here we use a more standard 4D layout
    # (B, S, H, D) which is easier to work with. The strides passed to the kernel handle the layout.
    q = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), device='cuda', dtype=torch.float32)
    k = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), device='cuda', dtype=torch.float32)
    v = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), device='cuda', dtype=torch.float32)

    # Run Triton implementation
    triton_output = mha(q, k, v)

    # Run PyTorch reference implementation
    # Note: The reference implementation can be slow due to Python loops.
    # We'll test on a smaller sequence length to make it faster.
    print("Running verification on a smaller tensor size to avoid long wait times...")
    q_small = q[:, :4, :, :].clone()
    k_small = k[:, :4, :, :].clone()
    v_small = v[:, :4, :, :].clone()
    
    triton_output_small = mha(q_small, k_small, v_small)
    torch_output_small = torch_reference_mha(q_small, k_small, v_small)

    # Compare results
    print(f"Triton output shape: {triton_output.shape}")
    if torch.allclose(triton_output_small, torch_output_small, atol=1e-5, rtol=1e-4):
        print("✅ Triton and PyTorch results match!")
    else:
        print("❌ Triton and PyTorch results DO NOT match!")
        # Print difference for debugging
        print("Max difference:", (triton_output_small - torch_output_small).abs().max().item())

'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/mha/triton/kernel.py
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
  File "/workspace/cu2til/cases/mha/triton/kernel.py", line 184, in <module>
    triton_output = mha(q, k, v)
                    ^^^^^^^^^^^^
  File "/workspace/cu2til/cases/mha/triton/kernel.py", line 116, in mha
    mha_triton_kernel[grid](
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
triton.compiler.errors.CompilationError: at 30:13:

    Grid: (BATCH_SIZE, SEQ_LEN)
    """
    # 1. Get program IDs to identify the current (batch, seq_pos)
    pid_batch = tl.program_id(axis=0)
    pid_seq = tl.program_id(axis=1)

    # 2. Load Q, K, V blocks for the current (batch, seq_pos)
    # We need to load all heads for this position. Shape: [NUM_HEADS, HEAD_DIM]

    # Offsets for the head and dimension axes
    offs_h = tl.arange(0, NUM_HEADS)
             ^
'''