import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attn_forward_kernel(
    # Input/Output Pointers
    Q, K, V, O,
    # Stride information for Tensors
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # Other metadata
    Z, H, N_CTX,
    # Kernel-specific constants
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Triton kernel for Flash Attention.
    This is a translation of the provided CUDA kernel, using tl.make_block_ptr.
    """
    # 1. Get Program and Block IDs
    # This thread block processes the `start_m`-th block of Q rows
    start_m = tl.program_id(0)
    # This thread block processes the `off_hz`-th head/batch combination
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # 2. Initialize pointers to Q, K, V
    # These are the base pointers for the current head and batch
    q_base_ptr = Q + off_z * stride_qz + off_h * stride_qh
    k_base_ptr = K + off_z * stride_kz + off_h * stride_kh
    v_base_ptr = V + off_z * stride_vz + off_h * stride_vh
    o_base_ptr = O + off_z * stride_oz + off_h * stride_oh

    # 3. Initialize accumulators for online softmax
    # Corresponds to `lane_block_row_max_old`, `lane_block_row_sum_old`, and `R_D` in CUDA
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # 4. Load the Q tile from Global Memory
    # This corresponds to the initial `cp.async` load of the Q tile into smem
    q_offset_m = start_m * BLOCK_M
    
    # Create block pointers for Q
    q_ptrs = tl.make_block_ptr(
        base=q_base_ptr,
        shape=(N_CTX, D_HEAD),
        strides=(stride_qm, stride_qk),
        offsets=(q_offset_m, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    # Load Q. The mask prevents out-of-bounds access for the last block.
    q = tl.load(q_ptrs, boundary_check=(0, 1))

    # 5. Main loop over K and V tiles
    # Corresponds to `for (int tile_K_seqlen = 0; ...)` in CUDA
    # The original CUDA code is not causal, so we iterate over the full sequence.
    for start_n in range(0, N_CTX, BLOCK_N):
        # --- Load K and V for the current block ---
        # Create block pointers for K and V
        k_ptrs = tl.make_block_ptr(
            base=k_base_ptr,
            shape=(D_HEAD, N_CTX), # Note: shape is (D_HEAD, N_CTX) for transposed matmul
            strides=(stride_kk, stride_kn),
            offsets=(0, start_n),
            block_shape=(D_HEAD, BLOCK_N),
            order=(0, 1)
        )
        v_ptrs = tl.make_block_ptr(
            base=v_base_ptr,
            shape=(N_CTX, D_HEAD),
            strides=(stride_vn, stride_vk),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, D_HEAD),
            order=(1, 0)
        )
        # Load K and V tiles. Boundary checks handle sequences not divisible by BLOCK_N.
        k = tl.load(k_ptrs, boundary_check=(1,))
        v = tl.load(v_ptrs, boundary_check=(0,))

        # --- Compute S = Q @ K^T ---
        # Corresponds to the first `mma.sync` loop in CUDA
        s_ij = tl.dot(q, k)
        # The CUDA code applies scale during the softmax step. Applying it here is equivalent.
        s_ij *= (1.0 / tl.sqrt(D_HEAD.to(tl.float32)))

        # --- Online Softmax Calculation ---
        # This block replaces the manual max reduction, exp, and sum reduction in CUDA
        
        # a. Find new max
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        # b. Calculate P_ij and rescale l_i, acc
        alpha = tl.exp(m_i - m_new)
        p_ij = tl.exp(s_ij - m_new[:, None])
        
        l_new = alpha * l_i + tl.sum(p_ij, axis=1)
        
        # c. Update accumulator
        # Rescale previous accumulator value
        acc *= alpha[:, None]
        # Add the contribution of the current tile: P_ij @ V
        # Corresponds to the second `mma.sync` loop in CUDA
        acc += tl.dot(p_ij.to(Q.dtype.element_ty), v)
        
        # d. Update m_i and l_i for the next iteration
        l_i = l_new
        m_i = m_new

    # 6. Final Rescaling and Store to Output
    # Corresponds to the final rescaling and store logic in CUDA
    # Normalize the accumulator
    acc = acc / l_i[:, None]

    # Create block pointer for the output tensor O
    o_ptrs = tl.make_block_ptr(
        base=o_base_ptr,
        shape=(N_CTX, D_HEAD),
        strides=(stride_om, stride_ok),
        offsets=(q_offset_m, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    # Store the final result.
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), boundary_check=(0, 1))


def flash_attn_triton(Q, K, V):
    """
    Python launcher for the Triton Flash Attention kernel.
    """
    # 1. Input validation
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.dtype == torch.float16 and K.dtype == torch.float16 and V.dtype == torch.float16
    assert Q.shape == K.shape == V.shape
    
    Z, H, N_CTX, D_HEAD = Q.shape
    
    # 2. Output tensor
    O = torch.empty_like(Q)

    # 3. Kernel configuration
    # These values are taken from the CUDA kernel's `Br` and `Bc` constants
    BLOCK_M = 64
    BLOCK_N = 32
    
    # The grid is 2D:
    # - dim 0: iterates over the Q sequence length
    # - dim 1: iterates over batches and heads
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    # 4. Launch kernel
    _flash_attn_forward_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        Z, H, N_CTX,
        D_HEAD=D_HEAD,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=D_HEAD, # BLOCK_DMODEL is an alias for D_HEAD
    )
    
    return O

# Example Usage
if __name__ == '__main__':
    Z, H, N_CTX, D_HEAD = 4, 12, 2048, 64

    # Create sample tensors
    q = torch.randn((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device='cuda')
    k = torch.randn((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device='cuda')
    v = torch.randn((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device='cuda')

    # Run Triton implementation
    triton_output = flash_attn_triton(q, k, v)

    # For verification, run a PyTorch equivalent (requires a lot of memory)
    try:
        pytorch_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        print("Comparing Triton output to PyTorch SDPA...")
        # Note: Due to different computation orders and precision, small differences are expected.
        # The tolerance `atol` might need adjustment based on the problem size.
        assert torch.allclose(triton_output, pytorch_output, atol=1e-1, rtol=1e-2)
        print("✅ Triton and PyTorch outputs match!")
    except torch.cuda.OutOfMemoryError:
        print("Could not run PyTorch reference due to OOM. Verification skipped.")
    except Exception as e:
        print(f"An error occurred during verification: {e}")

    print("\nTriton Output (first 4x4 of first head/batch):")
    print(triton_output[0, 0, :4, :4])
'''
root@ubuntu-ThinkStation-P520:/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some# python reducecode_direct_2_turn1.py 
Traceback (most recent call last):
  File "/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/reducecode_direct_2_turn1.py", line 187, in <module>
    triton_output = flash_attn_triton(q, k, v)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/reducecode_direct_2_turn1.py", line 162, in flash_attn_triton
    _flash_attn_forward_kernel[grid](
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
triton.compiler.errors.CompilationError: at 88:31:
            block_shape=(BLOCK_N, D_HEAD),
            order=(1, 0)
        )
        # Load K and V tiles. Boundary checks handle sequences not divisible by BLOCK_N.
        k = tl.load(k_ptrs, boundary_check=(1,))
        v = tl.load(v_ptrs, boundary_check=(0,))

        # --- Compute S = Q @ K^T ---
        # Corresponds to the first `mma.sync` loop in CUDA
        s_ij = tl.dot(q, k)
        # The CUDA code applies scale during the softmax step. Applying it here is equivalent.
        s_ij *= (1.0 / tl.sqrt(D_HEAD.to(tl.float32)))
                               ^
AttributeError("'int' object has no attribute 'to'")
'''