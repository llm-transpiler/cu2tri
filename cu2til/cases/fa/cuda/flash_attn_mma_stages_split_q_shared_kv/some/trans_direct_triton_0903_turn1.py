import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    sm_scale,
    seqlen_q,
    seqlen_k,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    STAGES: tl.constexpr,
):
    """
    Triton kernel for Flash Attention.
    This kernel is a translation of the provided CUDA C++ kernel.

    Grid: (cdiv(seqlen_q, BLOCK_M), batch * num_heads)
    """
    # 1. Get program IDs to identify the current work item
    start_m = tl.program_id(0)
    b_h_id = tl.program_id(1)
    
    # Unpack batch and head ID
    num_heads = stride_qz // stride_qh
    b_id = b_h_id // num_heads
    h_id = b_h_id % num_heads

    # 2. Initialize pointers and offsets
    # Create block-level offsets for the M (sequence Q) dimension
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Create block-level offsets for the D_HEAD dimension
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Pointers to the current block of Q, K, V, O
    # Offsets are calculated for the specific batch and head
    Q_block_ptr = Q_ptr + b_id * stride_qz + h_id * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    K_block_ptr = K_ptr + b_id * stride_kz + h_id * stride_kh + (offs_d[:, None] * stride_kk) # K is (D_HEAD, N)
    V_block_ptr = V_ptr + b_id * stride_vz + h_id * stride_vh + (offs_d[None, :] * stride_vk) # V is (N, D_HEAD)
    O_block_ptr = O_ptr + b_id * stride_oz + h_id * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)

    # 3. Initialize accumulators for the online softmax
    # This corresponds to R_D, lane_block_row_max_old, and lane_block_row_sum_old
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)

    # 4. Load the Q block, it's loop-invariant
    # This corresponds to the initial `cp.async` load of the Q tile into smem
    q = tl.load(Q_block_ptr, mask=offs_m[:, None] < seqlen_q, other=0.0)

    # 5. Main loop over blocks of K and V
    # This corresponds to `for (int tile_K_seqlen = 0; ...)`
    # The STAGES parameter enables software pipelining, similar to the kStage=2 logic
    for start_n in range(0, seqlen_k, BLOCK_N):
        # -- Load K and V for the current block --
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Load K block: (BLOCK_DMODEL, BLOCK_N)
        k = tl.load(K_block_ptr + offs_n[None, :] * stride_kn, mask=offs_n[None, :] < seqlen_k, other=0.0)
        
        # Load V block: (BLOCK_N, BLOCK_DMODEL)
        v = tl.load(V_block_ptr + offs_n[:, None] * stride_vn, mask=offs_n[:, None] < seqlen_k, other=0.0)

        # -- Compute S = Q @ K.T --
        # This replaces the inner loop over kHeadDim and the `mma.sync` for S
        s_ij = tl.dot(q, k, out_dtype=tl.float32) * sm_scale

        # -- Online Softmax Calculation --
        # This section mirrors the logic of computing max, rescaling, and summing
        
        # 1. Find the new max for the combined blocks
        m_ij = tl.max(s_ij, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        # 2. Rescale previous accumulator (acc) and sum (l_i)
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        # 3. Calculate P_ij = exp(S_ij - m_new) and update sum l_i
        p_ij = tl.exp(s_ij - m_new[:, None])
        l_i += tl.sum(p_ij, 1)
        
        # 4. Update the max accumulator
        m_i = m_new

        # -- Compute O = P @ V and update accumulator --
        # Cast P to the correct dtype before the matmul, as in the CUDA kernel
        p_ij = p_ij.to(Q_ptr.dtype.element_ty)
        # This replaces the `mma.sync` for O
        acc += tl.dot(p_ij, v)

    # 6. Final rescaling and storing the output
    # This corresponds to the final block of the CUDA kernel
    l_i_reciprocal = 1.0 / l_i
    # Rescale the final output accumulator
    acc = acc * l_i_reciprocal[:, None]

    # Write the final result to global memory
    tl.store(O_block_ptr, acc.to(O_ptr.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)


def flash_attention(q, k, v):
    """
    Python wrapper for the Triton Flash Attention kernel.
    Args:
        q (torch.Tensor): Query tensor of shape (batch, num_heads, seqlen_q, d_head)
        k (torch.Tensor): Key tensor of shape (batch, num_heads, seqlen_k, d_head)
        v (torch.Tensor): Value tensor of shape (batch, num_heads, seqlen_k, d_head)
    Returns:
        torch.Tensor: Output tensor of shape (batch, num_heads, seqlen_q, d_head)
    """
    # Ensure tensors are on CUDA and contiguous
    assert all(t.is_cuda and t.is_contiguous() for t in (q, k, v))
    
    # Get tensor dimensions
    batch, num_heads, seqlen_q, d_head = q.shape
    _, _, seqlen_k, _ = k.shape

    # Create output tensor
    o = torch.empty_like(q)

    # Kernel configuration
    # These values are chosen to match the CUDA kernel's configuration
    # Br = 64 -> BLOCK_M = 64
    # Bc = 32 -> BLOCK_N = 32
    BLOCK_M = 64
    BLOCK_N = 32
    
    # The CUDA kernel has a switch for d_head. Triton can handle this generically.
    # We pass it as a compile-time constant.
    BLOCK_DMODEL = d_head

    # Set up the launch grid
    grid = (triton.cdiv(seqlen_q, BLOCK_M), batch * num_heads)

    # Scaling factor for attention
    sm_scale = 1.0 / (d_head ** 0.5)

    # Determine STAGES to match the CUDA kernel's pipelining (kStage=2)
    # On newer GPUs (Ampere+), 3 or 4 can be better. 2 is a faithful translation.
    STAGES = 2

    # Launch the kernel
    _flash_attention_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        sm_scale,
        seqlen_q,
        seqlen_k,
        D_HEAD=d_head,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        STAGES=STAGES,
        num_warps=4, # 4 warps * 32 threads/warp = 128 threads, matching the CUDA kernel
    )
    return o

# --- Verification ---
if __name__ == "__main__":
    # Test case matching the constraints of the original kernel
    BATCH, N_HEADS, SEQLEN, D_HEAD = 4, 12, 1024, 64

    # The original kernel asserts `QKV_seqlen % Bc == 0`, so SEQLEN must be a multiple of 32.
    assert SEQLEN % 32 == 0

    # Create random tensors
    q = torch.randn((BATCH, N_HEADS, SEQLEN, D_HEAD), dtype=torch.float16, device="cuda")
    k = torch.randn((BATCH, N_HEADS, SEQLEN, D_HEAD), dtype=torch.float16, device="cuda")
    v = torch.randn((BATCH, N_HEADS, SEQLEN, D_HEAD), dtype=torch.float16, device="cuda")

    # Run Triton implementation
    triton_output = flash_attention(q, k, v)

    # Run PyTorch reference implementation for verification
    # Note: PyTorch's built-in scaled_dot_product_attention is highly optimized and might use
    # its own Flash Attention implementation under the hood.
    torch_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

    # Compare results
    print(f"Triton output shape: {triton_output.shape}")
    print(f"PyTorch output shape: {torch_output.shape}")

    # The numerical results should be very close.
    # Using a slightly higher tolerance due to different computation orders and float precision.
    is_close = torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2)
    print(f"Outputs are close: {is_close}")

    # Benchmark
    print("\n--- Benchmarking ---")
    ms = triton.testing.do_bench(lambda: flash_attention(q, k, v))
    print(f"Triton implementation: {ms:.3f} ms")
    ms = triton.testing.do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False))
    print(f"PyTorch implementation: {ms:.3f} ms")
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/trans_direct_triton_0903_turn1.py
Triton output shape: torch.Size([4, 12, 1024, 64])
PyTorch output shape: torch.Size([4, 12, 1024, 64])
Outputs are close: True

--- Benchmarking ---
Triton implementation: 0.098 ms
PyTorch implementation: 0.097 ms
'''