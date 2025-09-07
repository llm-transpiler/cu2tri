import torch
import triton
import triton.language as tl
import math

# The Triton kernel that replaces the CUDA code
@triton.jit
def mha_kernel_triton(
    # Pointers to tensors
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # Strides for memory access
    stride_q_batch, stride_q_seq, stride_q_head,
    stride_k_batch, stride_k_seq, stride_k_head,
    stride_v_batch, stride_v_seq, stride_v_head,
    stride_o_batch, stride_o_seq, stride_o_head,
    # Compile-time constants
    NUM_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    """
    Triton kernel for a non-standard MHA that attends over heads for a single token.
    Each program instance computes the output for one token in the sequence.
    """
    # 1. Get program IDs to identify the current token
    pid_batch = tl.program_id(axis=0)
    pid_seq = tl.program_id(axis=1)

    # 2. Define offsets for loading the full (NUM_HEADS, D_HEAD) matrices for Q, K, V
    # These represent the indices for the head dimension and the feature dimension
    offs_head = tl.arange(0, NUM_HEADS)
    offs_d = tl.arange(0, D_HEAD)

    # 3. Create pointers to the start of the data for the current token
    q_token_ptr = Q_ptr + (pid_batch * stride_q_batch) + (pid_seq * stride_q_seq)
    k_token_ptr = K_ptr + (pid_batch * stride_k_batch) + (pid_seq * stride_k_seq)
    v_token_ptr = V_ptr + (pid_batch * stride_v_batch) + (pid_seq * stride_v_seq)

    # 4. Load Q, K, and V for the current token.
    # We load the entire (NUM_HEADS, D_HEAD) block for this token.
    # The shape of q_ptrs, k_ptrs, v_ptrs will be (NUM_HEADS, D_HEAD)
    q_ptrs = q_token_ptr + offs_head[:, None] * stride_q_head + offs_d[None, :]
    k_ptrs = k_token_ptr + offs_head[:, None] * stride_k_head + offs_d[None, :]
    v_ptrs = v_token_ptr + offs_head[:, None] * stride_v_head + offs_d[None, :]
    
    # Load the data into SRAM
    q = tl.load(q_ptrs) # shape: (NUM_HEADS, D_HEAD)
    k = tl.load(k_ptrs) # shape: (NUM_HEADS, D_HEAD)
    v = tl.load(v_ptrs) # shape: (NUM_HEADS, D_HEAD)

    # 5. Compute Q @ K.T
    # q is (NUM_HEADS, D_HEAD), k.T is (D_HEAD, NUM_HEADS)
    # The result `scores` is a (NUM_HEADS, NUM_HEADS) block.
    # This is much more efficient than the nested loops in CUDA.
    scores = tl.dot(q, tl.trans(k))

    # 6. Apply scaling factor
    scaling_factor = 1.0 / tl.sqrt(D_HEAD.to(tl.float32))
    scores *= scaling_factor

    # 7. Apply row-wise softmax
    # This is a standard and numerically stable softmax implementation in Triton
    row_max = tl.max(scores, axis=1)
    scores = scores - row_max[:, None]  # Subtract max for stability
    p = tl.exp(scores.to(tl.float32))
    row_sum = tl.sum(p, axis=1)
    p = p / row_sum[:, None] # p is now the softmax-ed attention matrix

    # 8. Compute the final output: P @ V
    # p is (NUM_HEADS, NUM_HEADS), v is (NUM_HEADS, D_HEAD)
    # The result `output` is a (NUM_HEADS, D_HEAD) block.
    output = tl.dot(p.to(q.dtype), v) # Ensure p is same dtype as v for dot product

    # 9. Write the output block back to global memory
    out_token_ptr = Out_ptr + (pid_batch * stride_o_batch) + (pid_seq * stride_o_seq)
    out_ptrs = out_token_ptr + offs_head[:, None] * stride_o_head + offs_d[None, :]
    tl.store(out_ptrs, output)


# Host wrapper function to launch the Triton kernel
def mha_triton_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for the non-standard MHA Triton kernel.
    
    Args:
        q, k, v: Input tensors of shape (BATCH, SEQ_LEN, NUM_HEADS, D_HEAD)
    
    Returns:
        Output tensor of the same shape.
    """
    # Ensure inputs are on the correct device and have the expected shape
    assert q.shape == k.shape == v.shape, "Q, K, and V must have the same shape"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Input tensors must be on a CUDA device"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Input tensors must be contiguous"

    batch_size, seq_len, num_heads, head_dim = q.shape

    # Create an empty output tensor
    output = torch.empty_like(q)

    # The grid defines how many instances of the kernel we launch.
    # We launch one instance per token in the batch.
    grid = (batch_size, seq_len)

    # Launch the kernel
    mha_kernel_triton[grid](
        q, k, v, output,
        # Strides for Q
        q.stride(0), q.stride(1), q.stride(2),
        # Strides for K
        k.stride(0), k.stride(1), k.stride(2),
        # Strides for V
        v.stride(0), v.stride(1), v.stride(2),
        # Strides for Output
        output.stride(0), output.stride(1), output.stride(2),
        # Constants
        NUM_HEADS=num_heads,
        D_HEAD=head_dim,
    )
    
    return output

# --- Verification ---
if __name__ == "__main__":
    # Parameters from the CUDA code
    BATCH_SIZE = 4
    SEQ_LEN = 2048
    NUM_HEADS = 6
    D_HEAD = 256

    # Use smaller dimensions for faster testing
    # BATCH_SIZE = 2
    # SEQ_LEN = 128
    # NUM_HEADS = 6
    # D_HEAD = 256

    print(f"Testing with: BATCH={BATCH_SIZE}, SEQ_LEN={SEQ_LEN}, HEADS={NUM_HEADS}, D_HEAD={D_HEAD}")

    # Create random input tensors
    torch.manual_seed(0)
    q = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, D_HEAD), device='cuda', dtype=torch.float32)
    k = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, D_HEAD), device='cuda', dtype=torch.float32)
    v = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, D_HEAD), device='cuda', dtype=torch.float32)

    # --- PyTorch reference implementation ---
    def mha_pytorch_reference(q, k, v):
        # The operation is performed independently for each token (batch, seq)
        # So we can reshape to combine those dims
        q_r = q.view(-1, NUM_HEADS, D_HEAD)
        k_r = k.view(-1, NUM_HEADS, D_HEAD)
        v_r = v.view(-1, NUM_HEADS, D_HEAD)

        # 1. Q @ K.T
        scores = torch.matmul(q_r, k_r.transpose(-2, -1))
        
        # 2. Scaling
        scores = scores / math.sqrt(D_HEAD)
        
        # 3. Softmax
        p = torch.nn.functional.softmax(scores, dim=-1)
        
        # 4. P @ V
        output_r = torch.matmul(p, v_r)
        
        # Reshape back to original dimensions
        return output_r.view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, D_HEAD)

    # Run both implementations
    output_pytorch = mha_pytorch_reference(q, k, v)
    output_triton = mha_triton_wrapper(q, k, v)

    # Compare the results
    print("Comparing PyTorch and Triton results...")
    is_close = torch.allclose(output_pytorch, output_triton, atol=1e-4, rtol=1e-4)
    print(f"Results are close: {is_close}")

    # Benchmark performance
    @triton.testing.perf_report(
        [triton.testing.Benchmark(
            x_names=['SEQ_LEN'],
            x_vals=[128, 256, 512, 1024, 2048, 4096],
            line_arg='provider',
            line_vals=['pytorch', 'triton'],
            line_names=['PyTorch', 'Triton'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='ms',
            plot_name=f'mha-head-attention-B{BATCH_SIZE}-H{NUM_HEADS}-D{D_HEAD}',
            args={'BATCH_SIZE': BATCH_SIZE, 'NUM_HEADS': NUM_HEADS, 'D_HEAD': D_HEAD},
        )]
    )
    def benchmark(BATCH_SIZE, SEQ_LEN, NUM_HEADS, D_HEAD, provider):
        q = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, D_HEAD), device='cuda', dtype=torch.float32)
        k = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, D_HEAD), device='cuda', dtype=torch.float32)
        v = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, D_HEAD), device='cuda', dtype=torch.float32)
        
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'pytorch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: mha_pytorch_reference(q, k, v), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: mha_triton_wrapper(q, k, v), quantiles=quantiles)
        return ms, min_ms, max_ms

    print("\nRunning benchmark...")
    benchmark.run(show_plots=True, print_data=True)
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/mha_1_2048_6_256/triton/api_chat_5/kernel_turn1.py
Testing with: BATCH=4, SEQ_LEN=2048, HEADS=6, D_HEAD=256
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
  File "/workspace/cu2til/cases/mha_1_2048_6_256/triton/api_chat_5/kernel_turn1.py", line 169, in <module>
    output_triton = mha_triton_wrapper(q, k, v)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/cu2til/cases/mha_1_2048_6_256/triton/api_chat_5/kernel_turn1.py", line 105, in mha_triton_wrapper
    mha_kernel_triton[grid](
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
triton.compiler.errors.CompilationError: at 23:16:
):
    """
    Triton kernel for a non-standard MHA that attends over heads for a single token.
    Each program instance computes the output for one token in the sequence.
    """
    # 1. Get program IDs to identify the current token
    pid_batch = tl.program_id(axis=0)
    pid_seq = tl.program_id(axis=1)

    # 2. Define offsets for loading the full (NUM_HEADS, D_HEAD) matrices for Q, K, V
    # These represent the indices for the head dimension and the feature dimension
    offs_head = tl.arange(0, NUM_HEADS)
                ^
'''