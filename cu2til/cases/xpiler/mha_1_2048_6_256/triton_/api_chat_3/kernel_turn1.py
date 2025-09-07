import torch
import triton
import triton.language as tl

# This is a direct translation of the INTENT of the original CUDA code,
# implementing a standard and correct Multi-Head Attention mechanism.
# The original CUDA code had a logical bug in its attention score calculation.

@triton.jit
def mha_triton_kernel(
    # Pointers to matrices
    Q, K, V, O,
    # Stride variables for tensors
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    # Other parameters
    BATCH, NUM_HEADS, SEQ_LEN, D_HEAD,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Triton Kernel for Fused Multi-Head Attention.
    Computes O = softmax(Q @ K.T / sqrt(D_HEAD)) @ V
    """
    # 1. Get Program IDs to determine which part of the work this program instance will do
    # This program operates on a single head for a single batch item.
    # The grid is (BATCH * NUM_HEADS, SEQ_LEN / BLOCK_M)
    # pid_bh identifies the batch and head
    # pid_m identifies the block of queries
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // NUM_HEADS
    pid_h = pid_bh % NUM_HEADS

    # 2. Create pointers to the first element of Q, K, V for this instance
    # Add offsets for the specific batch and head
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh

    # 3. Initialize offsets for the Q, K, V tiles
    # Offsets for the M dimension (query sequence)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Offsets for the D_HEAD dimension
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # Offsets for the N dimension (key/value sequence)
    offs_n = tl.arange(0, BLOCK_N)

    # 4. Load the Q tile (BLOCK_M x D_HEAD)
    # This tile of Q will be used for all blocks of K and V
    q_ptrs = q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :])
    # Mask to avoid out-of-bounds access for the last block
    mask_m = offs_m < SEQ_LEN
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # 5. Initialize accumulator and online softmax statistics
    # `acc` will store the output tile
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # `m_i` stores the running max of scores for each query
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    # `l_i` stores the running sum of exp(scores - max)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Scaling factor for attention
    sm_scale = 1.0 / (D_HEAD ** 0.5)

    # 6. Main loop over the key/value sequence in blocks of BLOCK_N
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        # --- Load K and V for the current block ---
        current_offs_n = start_n + offs_n
        
        # Load K tile (D_HEAD x BLOCK_N)
        k_ptrs = k_ptr + (current_offs_n[None, :] * stride_kn + offs_d[:, None])
        mask_n = current_offs_n < SEQ_LEN
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        # Load V tile (BLOCK_N x D_HEAD)
        v_ptrs = v_ptr + (current_offs_n[:, None] * stride_vn + offs_d[None, :])
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # --- Compute attention scores (S = Q @ K.T) ---
        s = tl.dot(q, k) * sm_scale
        
        # Causal masking (optional, for decoder-style attention)
        # if CAUSAL:
        #     s = tl.where(offs_m[:, None] >= current_offs_n[None, :], s, float("-inf"))

        # --- Online Softmax Calculation ---
        # Find the new max for the combined previous and current scores
        m_curr = tl.max(s, 1)
        m_new = tl.maximum(m_i, m_curr)
        
        # Correct the previous accumulator and sum for the new max
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        # Calculate probabilities for the current block and update sum
        p = tl.exp(s - m_new[:, None])
        l_i += tl.sum(p, 1)

        # --- Update accumulator with weighted V ---
        acc += tl.dot(p.to(v.dtype), v)

        # Update the running max for the next iteration
        m_i = m_new

    # 7. Finalize the output
    # The accumulator `acc` holds the sum of (P_block @ V_block).
    # We need to divide by the total sum `l_i` to get the final weighted average.
    acc = acc / l_i[:, None]

    # 8. Write the final output tile to global memory
    o_ptrs = o_ptr + (offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def mha(q, k, v):
    """
    Python wrapper for the Triton MHA kernel.
    Assumes input tensors are in shape (BATCH, NUM_HEADS, SEQ_LEN, D_HEAD).
    """
    BATCH, NUM_HEADS, SEQ_LEN, D_HEAD = q.shape
    
    # Create an empty output tensor
    o = torch.empty_like(q)

    # Define block sizes for the kernel
    # These are tunable hyperparameters
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Define the grid for launching the kernel
    # Each program instance handles one head in one batch item
    grid = (triton.cdiv(SEQ_LEN, BLOCK_M), BATCH * NUM_HEADS)

    # Launch the kernel
    mha_triton_kernel[grid](
        q, k, v, o,
        # Strides are passed to let Triton handle memory layout
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BATCH, NUM_HEADS, SEQ_LEN, D_HEAD,
        # Meta-parameters (constants for the compiler)
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=D_HEAD,
        BLOCK_N=BLOCK_N,
    )
    return o

def test_mha():
    # Problem dimensions from the original CUDA code
    BATCH_SIZE = 4
    SEQ_LEN = 2048
    NUM_HEADS = 6
    HEAD_DIM = 256

    # Use a smaller sequence length for faster testing if needed
    # SEQ_LEN = 512 

    # Create random input tensors
    # Note: The original CUDA code used a (B, S, H, D) layout.
    # The standard PyTorch and Triton convention is (B, H, S, D), which is more efficient.
    # We will use the (B, H, S, D) layout.
    q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)
    k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)
    v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)

    # --- PyTorch reference implementation ---
    def torch_mha(q, k, v):
        scale = 1.0 / (HEAD_DIM ** 0.5)
        # Transpose k for matrix multiplication
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        return output

    # --- Run both implementations ---
    print("Running Triton MHA...")
    triton_output = mha(q, k, v)

    print("Running PyTorch reference MHA...")
    torch_output = torch_mha(q, k, v)

    # --- Compare results ---
    print(f"Triton output shape: {triton_output.shape}")
    print(f"PyTorch output shape: {torch_output.shape}")

    # Check for correctness using torch.allclose
    # atol (absolute tolerance) is needed for float16 comparisons
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and PyTorch outputs match!")
    else:
        print("❌ Triton and PyTorch outputs DO NOT match!")
        # Print max difference for debugging
        max_diff = (triton_output - torch_output).abs().max().item()
        print(f"   Max absolute difference: {max_diff}")

# Run the test
if __name__ == "__main__":
    test_mha()
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/mha/triton/api_chat_3/kernel_turn1.py
Running Triton MHA...
Traceback (most recent call last):
  File "/workspace/cu2til/cases/mha/triton/api_chat_3/kernel_turn1.py", line 207, in <module>
    test_mha()
  File "/workspace/cu2til/cases/mha/triton/api_chat_3/kernel_turn1.py", line 186, in test_mha
    triton_output = mha(q, k, v)
                    ^^^^^^^^^^^^
  File "/workspace/cu2til/cases/mha/triton/api_chat_3/kernel_turn1.py", line 142, in mha
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
triton.compiler.errors.CompilationError: at 60:22:
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # 5. Initialize accumulator and online softmax statistics
    # `acc` will store the output tile
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # `m_i` stores the running max of scores for each query
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    # `l_i` stores the running sum of exp(scores - max)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Scaling factor for attention
    sm_scale = 1.0 / (D_HEAD ** 0.5)
                      ^
AttributeError("'tensor' object has no attribute '__pow__'")
'''