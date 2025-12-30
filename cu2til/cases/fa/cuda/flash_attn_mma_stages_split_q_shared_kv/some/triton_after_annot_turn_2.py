import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, # BLOCK_SIZE_K is an alias for HEAD_DIM in this context
):
    # 1. Get Program IDs and define offsets
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)

    # Early exit for blocks that are completely out of bounds for Q
    # This corresponds to `if (load_gmem_Q_Br >= QKV_seqlen) return;`
    start_m = pid_m * BLOCK_SIZE_M
    if start_m >= N_CTX:
        return

    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 2. Initialize pointers and load Q
    # Q is loop-invariant, so we load it once before the main loop.
    # This is a major performance optimization.
    q_ptrs = Q + pid_zh * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :]
    mask_q = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)

    # Pointers for K and V, to be used inside the loop
    # Note the shape of k_ptrs: we prepare it for a transposed load.
    k_ptrs = K + pid_zh * stride_kh + offs_k[:, None] * stride_kk + offs_n[None, :]
    v_ptrs = V + pid_zh * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :]
    
    # 3. Initialize accumulators for the online softmax
    # Corresponds to `R_D`, `lane_block_row_max_old`, `lane_block_row_sum_old`
    acc = tl.zeros([BLOCK_SIZE_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_SIZE_M], -float('inf'), dtype=tl.float32)
    
    scale = (HEAD_DIM ** -0.5)

    # 4. Main loop over the sequence length of K and V
    # This is the `serial_range_for(tile_K_seqlen, 0, Tc, 1)` loop in CUDA
    for start_n in range(0, N_CTX, BLOCK_SIZE_N):
        # -- Load K and V tiles --
        current_offs_n = start_n + offs_n
        mask_kv = current_offs_n[None, :] < N_CTX
        
        # Load K (transposed) and V
        k = tl.load(k_ptrs + start_n, mask=mask_kv, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn, mask=current_offs_n[:, None] < N_CTX, other=0.0)

        # -- Compute S = Q @ K.T --
        # Since k is loaded as (HEAD_DIM, BLOCK_SIZE_N), tl.dot(q, k) is Q @ K.T
        s_ij = tl.dot(q, k) * scale
        
        # =========================================================================
        # -- CORRECTED Online Softmax Calculation --
        # This is the numerically stable algorithm.
        # =========================================================================
        
        # 1. Find the new max for the combined statistics
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        # 2. Calculate weights for the update. `exp(m_old - m_new)` and `exp(m_block - m_new)`
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        
        # 3. Calculate the properly scaled probability block P_ij
        p_ij = beta[:, None] * tl.exp(s_ij - m_new[:, None])
        
        # 4. Update the row-sum normalizer `l`
        l_ij = tl.sum(p_ij, axis=1)
        l_new = alpha * l_i + l_ij
        
        # 5. Update the output accumulator `acc`
        # Rescale the old accumulator: acc_new = acc_old * exp(m_old - m_new)
        acc = acc * alpha[:, None]
        
        # Add the new value: acc_new += P_ij @ V_j
        p_ij_casted = p_ij.to(V.dtype.element_ty)
        acc += tl.dot(p_ij_casted, v)
        
        # 6. Update state for the next iteration
        l_i = l_new
        m_i = m_new

    # 5. Final Normalization and Store
    # The complex `warp_shuffle_spread` for storing is handled by `tl.store`.
    acc = acc / l_i[:, None]
    
    # Initialize output pointers and store the result
    o_ptrs = O + pid_zh * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_q)


def flash_attention(q, k, v):
    # Input validation
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == torch.float16 and k.dtype == torch.float16 and v.dtype == torch.float16

    # Tensor dimensions
    Z, H, N_CTX, HEAD_DIM = q.shape
    
    # Output tensor
    o = torch.empty_like(q)

    # Kernel configuration
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 64
    
    # Heuristics for number of warps
    num_warps = 4
    if HEAD_DIM >= 64:
        num_warps = 8
    
    grid = (triton.cdiv(N_CTX, BLOCK_SIZE_M), Z * H)

    # Launch the kernel
    _attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=HEAD_DIM, # Pass HEAD_DIM as BLOCK_SIZE_K
        num_warps=num_warps,
        num_stages=2 if HEAD_DIM >= 64 else 3, # Use pipelining
    )
    
    return o

# ================== Verification ==================
def test_flash_attention():
    Z, H, N_CTX, D_HEAD = 4, 12, 2048, 64
    dtype = torch.float16
    device = 'cuda'

    # Create random inputs
    q = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)

    # Run Triton implementation
    triton_output = flash_attention(q, k, v)

    # Run reference PyTorch implementation
    # Use float32 for the reference implementation's matmul for better precision
    p_ref = q @ k.transpose(-2, -1) / (D_HEAD**0.5)
    s_ref = torch.softmax(p_ref.float(), dim=-1).half()
    ref_output = s_ref @ v

    # Compare results
    print("Triton output shape:", triton_output.shape)
    print("Reference output shape:", ref_output.shape)

    # Check for correctness
    is_close = torch.allclose(triton_output, ref_output, atol=1e-2, rtol=0)
    print(f"Outputs are close: {is_close}")

    # Print max difference if not close
    if not is_close:
        max_diff = (triton_output - ref_output).abs().max().item()
        print(f"Max absolute difference: {max_diff}")

if __name__ == '__main__':
    test_flash_attention()

'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/triton_after_annot_turn_2.py
Triton output shape: torch.Size([4, 12, 2048, 64])
Reference output shape: torch.Size([4, 12, 2048, 64])
Outputs are close: False
/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/triton_after_annot_turn_2.py:178: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /opt/pytorch/pytorch/aten/src/ATen/native/Scalar.cpp:22.)
  max_diff = (triton_output - ref_output).abs().max().item()
Max absolute difference: 1.26953125
'''