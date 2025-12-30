import torch
import triton
import triton.language as tl

# This is the Triton equivalent of the CUDA kernel.
@triton.jit
def _flash_attention_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Triton kernel for Flash Attention.
    
    This kernel computes attention for a single head and a single batch item.
    The grid is structured as (num_blocks_m, num_heads * num_batches).
    
    - BLOCK_M: The block size for the Q sequence length dimension. Corresponds to `Br` in CUDA.
    - BLOCK_N: The block size for the K/V sequence length dimension. Corresponds to `Bc` in CUDA.
    - BLOCK_DMODEL: The head dimension. Corresponds to `kHeadDim` in CUDA.
    """
    # 1. Get Program IDs to determine which block of Q and which head/batch we are processing.
    pid_m = tl.program_id(0)  # ID for the M-dimension block (along Q's sequence length)
    pid_z = tl.program_id(1)  # ID for the batch/head dimension
    
    # Decompose pid_z into batch and head indices
    pid_batch = pid_z // H
    pid_head = pid_z % H

    # 2. Compute memory offsets for Q, K, V, O
    # Offsets for the current block of Q rows
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Offsets for the head dimension
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Pointers to the start of the Q, K, V, O tensors for the current batch and head
    q_base_ptr = Q + pid_batch * stride_qz + pid_head * stride_qh
    k_base_ptr = K + pid_batch * stride_kz + pid_head * stride_kh
    v_base_ptr = V + pid_batch * stride_vz + pid_head * stride_vh
    o_base_ptr = O + pid_batch * stride_oz + pid_head * stride_oh
    
    # Pointers to the specific tile of Q we will load
    # Shape: [BLOCK_M, BLOCK_DMODEL]
    q_ptrs = q_base_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    
    # 3. Initialize accumulators and statistics for online softmax
    # This is equivalent to R_D, lane_block_row_max_old, and lane_block_row_sum_old
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # 4. Load the Q tile from global memory to SRAM (compiler handles this)
    # This corresponds to the initial `cp.async` for Q in the CUDA code.
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # 5. Main loop over blocks of K and V
    # This corresponds to `for (int tile_K_seqlen = 0; ...)`
    for start_n in range(0, N_CTX, BLOCK_N):
        # Offsets for the current block of K/V rows
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # --- Load K and V for the current block ---
        # This corresponds to the pipelined `cp.async` for K and V
        # Pointers to the K tile. Shape: [BLOCK_N, BLOCK_DMODEL]
        k_ptrs = k_base_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        # Pointers to the V tile. Shape: [BLOCK_N, BLOCK_DMODEL]
        v_ptrs = v_base_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        
        # Load with masking for the last block
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        
        # --- Compute S = Q @ K.T ---
        # This corresponds to the MMA loop `for (int tile_K_d = ...)`
        # `tl.dot` abstracts away `ldmatrix` and `mma.sync`
        s_ij = tl.dot(q, k, trans_b=True) * sm_scale
        
        # --- Online Softmax Update ---
        # This block implements the logic for updating m_i, l_i, and the accumulator `acc`
        
        # a. Find the new max for the current row block
        m_ij = tl.max(s_ij, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # b. Rescale previous accumulator and update l_i
        alpha = tl.exp(m_i - m_i_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        
        # c. Compute P_ij = exp(S_ij - m_i_new)
        p_ij = tl.exp(s_ij - m_i_new[:, None])
        
        # d. Update l_i with the sum of the new P_ij values
        l_ij = tl.sum(p_ij, axis=1)
        l_i += l_ij
        
        # e. Update accumulator with the new values: acc += P_ij @ V
        # Cast p_ij to the input type for the dot product
        p_ij = p_ij.to(Q.dtype.element_ty)
        acc += tl.dot(p_ij, v)
        
        # f. Update the running max
        m_i = m_i_new

    # 6. Final normalization and store to global memory
    # This corresponds to the final rescaling and store operations in CUDA
    
    # Invert l_i to multiply instead of divide
    l_i_rcp = 1.0 / l_i
    # Normalize the accumulator
    acc = acc * l_i_rcp[:, None]
    
    # Pointers to the output block
    # Shape: [BLOCK_M, BLOCK_DMODEL]
    o_ptrs = o_base_ptr + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    
    # Write the final result to global memory
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


# Python launcher function, equivalent to the C++/Pybind11 wrapper
def flash_attn_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launcher for the Triton Flash Attention kernel.

    Args:
        q (torch.Tensor): Query tensor, shape (Z, H, N_CTX, D_HEAD)
        k (torch.Tensor): Key tensor, shape (Z, H, N_CTX, D_HEAD)
        v (torch.Tensor): Value tensor, shape (Z, H, N_CTX, D_HEAD)

    Returns:
        torch.Tensor: Output tensor, shape (Z, H, N_CTX, D_HEAD)
    """
    # 1. Validate inputs
    assert q.shape == k.shape == v.shape
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == torch.float16, "Only float16 is supported for now"
    
    Z, H, N_CTX, D_HEAD = q.shape
    
    # 2. Create output tensor
    o = torch.empty_like(q)
    
    # 3. Define block sizes. These can be tuned.
    # The values are chosen to match the CUDA kernel's configuration.
    # Br = 64 -> BLOCK_M = 64
    # Bc = 32 -> BLOCK_N = 32
    BLOCK_M = 64
    BLOCK_N = 32
    
    # The CUDA kernel supports D_HEAD in {32, 64, 96, 128}
    # Triton can handle this with a single kernel thanks to `constexpr`.
    
    # 4. Define the grid for launching the kernel
    # One program per block of Q's sequence dimension
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    
    # 5. Pre-compute scale factor
    sm_scale = 1.0 / (D_HEAD ** 0.5)
    
    # 6. Launch the kernel
    _flash_attention_kernel[grid](
        Q=q, K=k, V=v, O=o,
        stride_qz=q.stride(0), stride_qh=q.stride(1), stride_qm=q.stride(2), stride_qk=q.stride(3),
        stride_kz=k.stride(0), stride_kh=k.stride(1), stride_kn=k.stride(2), stride_kk=k.stride(3),
        stride_vz=v.stride(0), stride_vh=v.stride(1), stride_vn=v.stride(2), stride_vk=v.stride(3),
        stride_oz=o.stride(0), stride_oh=o.stride(1), stride_om=o.stride(2), stride_ok=o.stride(3),
        Z=Z, H=H, N_CTX=N_CTX,
        sm_scale=sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=D_HEAD,
    )
    
    return o

# Example Usage
if __name__ == '__main__':
    Z, H, N_CTX, D_HEAD = 4, 12, 1024, 64

    # Create random input tensors on GPU
    q = torch.randn((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device='cuda')
    k = torch.randn((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device='cuda')
    v = torch.randn((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device='cuda')

    # Run Triton implementation
    triton_output = flash_attn_triton(q, k, v)

    # For verification, run PyTorch's reference implementation
    # Note: PyTorch's SDP is highly optimized and might use a different backend.
    # This is for functional correctness checking.
    try:
        pytorch_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    except AttributeError:
        # Fallback for older PyTorch versions
        print("Using manual PyTorch implementation for verification.")
        sm_scale = 1.0 / (D_HEAD ** 0.5)
        S = (q @ k.transpose(-2, -1)) * sm_scale
        P = torch.nn.functional.softmax(S, dim=-1)
        pytorch_output = P @ v

    # Compare results
    print(f"Triton output shape: {triton_output.shape}")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    
    # Due to floating point arithmetic differences, we use allclose for comparison
    is_close = torch.allclose(triton_output, pytorch_output, atol=1e-2, rtol=1e-2)
    print(f"Outputs are close: {is_close}")
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/trans_direct_triton_0903_2_turn1.py
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 42, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
TypeError: dot() got an unexpected keyword argument 'trans_b'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/trans_direct_triton_0903_2_turn1.py", line 193, in <module>
    triton_output = flash_attn_triton(q, k, v)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/trans_direct_triton_0903_2_turn1.py", line 168, in flash_attn_triton
    _flash_attention_kernel[grid](
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
triton.compiler.errors.CompilationError: at 76:15:
        k_ptrs = k_base_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        # Pointers to the V tile. Shape: [BLOCK_N, BLOCK_DMODEL]
        v_ptrs = v_base_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

        # Load with masking for the last block
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # --- Compute S = Q @ K.T ---
        # This corresponds to the MMA loop `for (int tile_K_d = ...)`
        # `tl.dot` abstracts away `ldmatrix` and `mma.sync`
        s_ij = tl.dot(q, k, trans_b=True) * sm_scale
'''