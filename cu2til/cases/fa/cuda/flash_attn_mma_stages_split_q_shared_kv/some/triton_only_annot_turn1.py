import torch
import triton
import triton.language as tl

@triton.jit
def flash_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    seq_len,
    scale,
    # --- Block size constants passed from the launcher ---
    # These correspond to the template parameters in the CUDA code.
    # Br = BLOCK_M, Bc = BLOCK_N, kHeadDim = BLOCK_DMODEL
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Triton implementation of the flash_attn_mma_stages_split_q_shared_kv_kernel.
    """
    # =================================================================
    # 1. Get Program IDs and Calculate Initial Memory Pointers
    # =================================================================
    # Corresponds to block_id_x, block_id_y, QKV_batch_id, QKV_head_id
    start_m = tl.program_id(axis=0)
    batch_head_id = tl.program_id(axis=1)
    
    # In this layout, we assume the grid is (num_m_blocks, num_batches * num_heads)
    # For simplicity, we'll assume a flat batch*head dimension.
    # A more robust implementation might take separate batch and head IDs.

    # Calculate pointers for the first tile of Q, K, V, O
    # Offsets for the current block
    q_offset = batch_head_id * stride_qh + (start_m * BLOCK_M) * stride_qm
    k_offset = batch_head_id * stride_kh
    v_offset = batch_head_id * stride_vh
    o_offset = batch_head_id * stride_oh + (start_m * BLOCK_M) * stride_om

    Q_ptr += q_offset
    K_ptr += k_offset
    V_ptr += v_offset
    O_ptr += o_offset

    # =================================================================
    # 2. Initialize Accumulators for Online Softmax
    # =================================================================
    # Corresponds to lane_block_row_max_old, lane_block_row_sum_old, and R_D
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)

    # =================================================================
    # 3. Load the Q tile (once per block)
    # =================================================================
    # Corresponds to the first "COPY: G2S" for Q
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    q_ptrs = Q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    
    # Masking for the final block if seq_len is not a multiple of BLOCK_M
    q_mask = (start_m * BLOCK_M + offs_m) < seq_len
    
    # Load Q tile from global memory into registers (SRAM)
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # =================================================================
    # 4. Main Loop over K and V tiles
    # =================================================================
    # Corresponds to `serial_range_for(tile_K_seqlen, 0, Tc, 1)`
    # The loop iterates over the sequence length dimension of K and V.
    # Triton's compiler will pipeline the loads and computes in this loop.
    
    # Tc = seq_len / BLOCK_N
    for k_block_start in range(0, seq_len, BLOCK_N):
        offs_n = tl.arange(0, BLOCK_N)
        
        # --- Load K and V for the current block ---
        # Corresponds to "COPY: G2S" for K and V inside the loop
        k_ptrs = K_ptr + ((k_block_start + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        v_ptrs = V_ptr + ((k_block_start + offs_n)[None, :] * stride_vn + offs_d[:, None] * stride_vk)

        # Masking for the final block of K/V
        kv_mask = (k_block_start + offs_n) < seq_len
        
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask[None, :], other=0.0)

        # --- Compute S = Q @ K.T ---
        # Corresponds to the first GEMM `s_1 = q_1 @ k_1.T`
        # tl.dot automatically uses tensor cores (MMA instructions)
        s_ij = tl.dot(q, tl.trans(k)) * scale
        
        # Apply causal mask by setting scores for future tokens to -inf
        # This is a common addition for decoder-style attention, not explicitly
        # in the original snippet but essential for correctness in many cases.
        # We can add it here for completeness.
        # causal_mask = (start_m * BLOCK_M + offs_m[:, None]) >= (k_block_start + offs_n[None, :])
        # s_ij = tl.where(causal_mask, s_ij, float('-inf'))

        # --- Online Softmax Update ---
        # Corresponds to all the `COMPUTE` blocks for max, sum, exp, etc.
        
        # 1. Find the new row-wise maximum of the current score block
        m_ij = tl.max(s_ij, axis=1)
        
        # 2. Find the new overall maximum
        m_new = tl.maximum(m_i, m_ij)
        
        # 3. Calculate scaling factors for the update
        # exp(m_i - m_new)
        alpha = tl.exp(m_i - m_new)
        # exp(s_ij - m_new)
        p_ij = tl.exp(s_ij - m_new[:, None])
        
        # 4. Update the row-wise sum of exponents (the softmax denominator)
        # l_new = sum(p_ij)
        l_new = tl.sum(p_ij, axis=1)
        # l_i_new = alpha * l_i + l_new
        l_i_new = alpha * l_i + l_new
        
        # 5. Update the accumulator
        # acc = acc * alpha + (p_ij @ V)
        # Rescale the old accumulator
        acc = acc * alpha[:, None]
        
        # --- Compute O_partial = P @ V ---
        # Corresponds to the second GEMM `o_1 = s_1 @ v_1`
        # We need to cast p_ij to the same dtype as v for the dot product
        p_ij = p_ij.to(v.dtype)
        o_partial = tl.dot(p_ij, v)
        
        # Add the new partial output to the accumulator
        acc += o_partial
        
        # 6. Update statistics for the next iteration
        m_i = m_new
        l_i = l_i_new

    # =================================================================
    # 5. Final Normalization and Store to Global Memory
    # =================================================================
    # Corresponds to the final `RCP, MUL, CAST` and store operations
    
    # Normalize the final accumulator by the total sum of exponents
    # Add a small epsilon to avoid division by zero
    l_i_rcp = 1.0 / (l_i + 1e-10)
    
    # Final output is acc / l_i
    # The mask is applied to the output before storing
    acc = acc * l_i_rcp[:, None]
    
    # Create output pointers
    o_ptrs = O_ptr + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    
    # Store the final result to global memory
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_mask[:, None])


# --- Launcher function to run the Triton kernel ---
def flash_attention(q, k, v, scale):
    # q, k, v: (Z, H, N_CTX, D_HEAD) tensors
    # Z: batch_size, H: num_heads, N_CTX: seq_len, D_HEAD: head_dim
    
    Z, H, N_CTX, D_HEAD = q.shape
    
    # Create an empty output tensor
    o = torch.empty_like(q)
    
    # Define block sizes. These should be tuned for the specific GPU architecture.
    # The values are chosen to be similar to typical FlashAttention configurations.
    BLOCK_M = 128
    BLOCK_N = 64
    
    # The grid defines how many instances of the kernel will run.
    # Each instance (or "program") processes one BLOCK_M chunk of Q.
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    
    # Call the Triton kernel
    flash_attn_kernel[grid](
        q, k, v, o,
        # Strides for Q
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # Strides for K
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # Strides for V
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # Strides for O
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        seq_len=N_CTX,
        scale=scale,
        # Pass block sizes as constexpr arguments
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=D_HEAD,
        # On newer Triton versions, num_warps can be specified for performance tuning
        num_warps=4,
        # num_stages > 1 enables software pipelining, similar to the CUDA code's intent
        num_stages=2, 
    )
    
    return o

# Example Usage:
if __name__ == '__main__':
    Z, H, N_CTX, D_HEAD = 4, 12, 2048, 64
    
    # Use float16 for performance, as intended by the original kernel
    dtype = torch.float16
    device = 'cuda'

    q = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device)
    k = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device)
    v = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device)
    scale = 1.0 / (D_HEAD ** 0.5)

    # Run Triton implementation
    triton_output = flash_attention(q, k, v, scale)

    # For verification, run PyTorch's reference implementation
    # Note: PyTorch's native implementation is highly optimized and might be faster.
    # This is for correctness checking.
    try:
        # Use the highly optimized backend if available
        from torch.nn.functional import scaled_dot_product_attention
        pytorch_output = scaled_dot_product_attention(q, k, v, scale=scale)
    except ImportError:
        # Fallback to a manual implementation if the above is not available
        print("Using manual PyTorch implementation for verification.")
        p = torch.matmul(q, k.transpose(2, 3)) * scale
        p = torch.softmax(p.float(), dim=-1).half()
        pytorch_output = torch.matmul(p, v)

    # Compare results
    print(f"Triton output shape: {triton_output.shape}")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    
    # The tolerance for float16 can be higher than for float32
    print(f"max diff: {torch.abs(triton_output - pytorch_output).max()}")
    print(f"mean diff: {torch.abs(triton_output - pytorch_output).mean()}")
    print(f"value range: {triton_output.min()}, {triton_output.max()}")
    print(f"ref value range: {pytorch_output.min()}, {pytorch_output.max()}")
    is_close = torch.allclose(triton_output, pytorch_output, atol=1e-2, rtol=1e-2)
    print(f"Outputs are close: {is_close}")
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/triton_only_annot_turn1.py
Triton output shape: torch.Size([4, 12, 2048, 64])
PyTorch output shape: torch.Size([4, 12, 2048, 64])
max diff: 0.49365234375
mean diff: 0.04052734375
value range: -0.276611328125, 0.3046875
ref value range: -0.32373046875, 0.31298828125
Outputs are close: False
'''