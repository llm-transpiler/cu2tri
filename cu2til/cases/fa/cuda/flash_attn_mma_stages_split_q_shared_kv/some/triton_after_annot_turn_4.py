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
):
    # 1. Get Program IDs and define offsets
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)

    start_m = pid_m * BLOCK_SIZE_M
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for Q, K, V
    # This q_mask is critical for correctness.
    mask_q = offs_m[:, None] < N_CTX
    
    # Early exit for blocks that are completely out of bounds for Q
    if start_m >= N_CTX:
        return

    # 2. Initialize pointers and load Q
    q_ptrs = Q + pid_zh * stride_qh + offs_m[:, None] * stride_qm + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)

    # Pointers for K and V
    k_ptrs = K + pid_zh * stride_kh + tl.arange(0, HEAD_DIM)[:, None] * stride_kk + offs_n[None, :]
    v_ptrs = V + pid_zh * stride_vh + offs_n[:, None] * stride_vn + tl.arange(0, HEAD_DIM)[None, :]
    
    # 3. Initialize accumulators
    acc = tl.zeros([BLOCK_SIZE_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_SIZE_M], -float('inf'), dtype=tl.float32)
    
    scale = (HEAD_DIM ** -0.5)

    # 4. Main loop over K and V
    for start_n in range(0, N_CTX, BLOCK_SIZE_N):
        current_offs_n = start_n + offs_n
        mask_k = current_offs_n[None, :] < N_CTX
        
        # -- Load K and V tiles --
        k = tl.load(k_ptrs + start_n, mask=mask_k, other=0.0)
        
        mask_v = current_offs_n[:, None] < N_CTX
        v = tl.load(v_ptrs + start_n * stride_vn, mask=mask_v, other=0.0)

        # -- Compute S = Q @ K.T --
        s_ij = tl.dot(q, k) * scale
        
        # =========================================================================
        # THE FINAL, CORRECT MASKING LOGIC
        # We must combine the mask from Q's rows and K's columns.
        # If a Q row is out of bounds, all its scores must be -inf.
        # If a K column is out of bounds, its scores must be -inf.
        # The combined_mask handles both cases.
        # =========================================================================
        combined_mask = mask_q & mask_k
        s_ij = tl.where(combined_mask, s_ij, -float('inf'))

        # -- Online Softmax Calculation --
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p_ij = tl.exp(s_ij - m_new[:, None])
        l_ij = tl.sum(p_ij, axis=1)
        l_new = alpha * l_i + l_ij
        
        # -- Update output accumulator `acc` --
        acc = acc * alpha[:, None]
        p_ij_casted = p_ij.to(V.dtype.element_ty)
        acc += tl.dot(p_ij_casted, v)
        
        # -- Update state for the next iteration --
        l_i = l_new
        m_i = m_new

    # 5. Final Normalization and Store
    acc = acc / l_i[:, None]
    
    o_ptrs = O + pid_zh * stride_oh + offs_m[:, None] * stride_om + tl.arange(0, HEAD_DIM)[None, :]
    # Use the original q_mask to prevent writing to out-of-bounds memory
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_q)


def flash_attention(q, k, v):
    Z, H, N_CTX, HEAD_DIM = q.shape
    o = torch.empty_like(q)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 64
    
    num_warps = 4
    if HEAD_DIM >= 64:
        num_warps = 8
    
    grid = (triton.cdiv(N_CTX, BLOCK_SIZE_M), Z * H)

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
        num_warps=num_warps,
        num_stages=2 if HEAD_DIM >= 64 else 3,
    )
    
    return o

# ================== Verification ==================
def test_flash_attention(N_CTX_override=None):
    Z, H, N_CTX_orig, D_HEAD = 4, 12, 2048, 64
    N_CTX = N_CTX_override if N_CTX_override is not None else N_CTX_orig
    
    print(f"--- Testing with N_CTX = {N_CTX} ---")

    dtype = torch.float16
    device = 'cuda'

    torch.manual_seed(0)
    q_full = torch.randn((Z, H, N_CTX_orig, D_HEAD), dtype=dtype, device=device)
    k_full = torch.randn((Z, H, N_CTX_orig, D_HEAD), dtype=dtype, device=device)
    v_full = torch.randn((Z, H, N_CTX_orig, D_HEAD), dtype=dtype, device=device)

    # Use slices to test non-multiple sequence lengths
    q = q_full[:, :, :N_CTX, :]
    k = k_full[:, :, :N_CTX, :]
    v = v_full[:, :, :N_CTX, :]

    # Run Triton implementation
    triton_output = flash_attention(q, k, v)

    # Run reference PyTorch implementation
    p_ref = q.float() @ k.transpose(-2, -1).float() / (D_HEAD**0.5)
    s_ref = torch.softmax(p_ref, dim=-1).half()
    ref_output = s_ref @ v

    # Compare results
    print("Triton output shape:", triton_output.shape)
    print("Reference output shape:", ref_output.shape)

    is_close = torch.allclose(triton_output, ref_output, atol=1e-2, rtol=0)
    print(f"Outputs are close: {is_close}")

    if not is_close:
        max_diff = (triton_output - ref_output).abs().max().item()
        print(f"Max absolute difference: {max_diff}")
    print("-" * 30)

if __name__ == '__main__':
    # Test case where N_CTX is a multiple of block size
    test_flash_attention(N_CTX_override=2048)
    # Test case where N_CTX is NOT a multiple of block size (critical for testing masking)
    test_flash_attention(N_CTX_override=2000)

'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/triton_after_annot_turn_3.py
Triton output shape: torch.Size([4, 12, 2048, 64])
Reference output shape: torch.Size([4, 12, 2048, 64])
Outputs are close: False
Max absolute difference: 0.4755859375
'''