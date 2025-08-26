import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attn_triton_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    batch_size, num_heads, seq_len,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Triton kernel for Flash Attention.

    This kernel computes attention for a single head and a single batch item.
    The grid is structured as (cdiv(seq_len, BLOCK_M), batch_size * num_heads).
    """
    # 1. Get Program IDs to determine which block of the problem we are responsible for.
    start_m = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    
    # Unpack batch and head indices
    batch_id = batch_head_id // num_heads
    head_id = batch_head_id % num_heads

    # 2. Initialize pointers and offsets
    # Create tile-sized ranges for the M, N, and DMODEL dimensions.
    # These are the offsets for the threads within this program instance.
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Pointers to the start of the Q, K, V matrices for this specific batch and head.
    q_ptr = Q + batch_id * stride_qz + head_id * stride_qh
    k_ptr = K + batch_id * stride_kz + head_id * stride_kh
    v_ptr = V + batch_id * stride_vz + head_id * stride_vh
    o_ptr = O + batch_id * stride_oz + head_id * stride_oh

    # 3. Initialize accumulators for the online softmax algorithm.
    # `acc` will store the numerator of the attention formula (P_ij * V_j).
    # `m_i` stores the running maximum of the dot products for numerical stability.
    # `l_i` stores the running sum of the exponentiated dot products (the denominator).
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # 4. Load the Q block. This block is stationary for the entire loop.
    # Create pointers for the Q block.
    q_ptrs = q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    
    # Create a mask to avoid loading out-of-bounds data for sequences not perfectly
    # divisible by BLOCK_M.
    m_mask = offs_m < seq_len
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    # 5. Main loop over blocks of K and V.
    # The loop iterates from 0 to seq_len with a step of BLOCK_N.
    for start_n in range(0, seq_len, BLOCK_N):
        # -- Load K and V for the current block --
        current_offs_n = start_n + offs_n
        n_mask = current_offs_n < seq_len

        # Pointers for K (transposed access) and V (standard access)
        k_ptrs = k_ptr + (offs_d[:, None] * stride_kk + current_offs_n[None, :] * stride_kn)
        v_ptrs = v_ptr + (current_offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

        k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # -- Compute S = Q * K^T --
        s_ij = tl.dot(q, k) * scale

        # Mask out-of-bounds scores for causal attention or padding.
        # Here, we only handle padding. For causal, you'd add: `s_ij = tl.where(offs_m[:, None] >= current_offs_n[None, :], s_ij, -float('inf'))`
        s_ij = tl.where(m_mask[:, None] & n_mask[None, :], s_ij, -float('inf'))

        # -- Online Softmax Update --
        # Find the new maximum for the current block.
        m_i_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        
        # Calculate the scaled probabilities P_ij for the current block.
        p_ij = tl.exp(s_ij - m_i_new[:, None])
        
        # Calculate the rescaling factor for the old accumulator and sum.
        alpha = tl.exp(m_i - m_i_new)
        
        # Update the sum (denominator `l_i`).
        l_i_new = alpha * l_i + tl.sum(p_ij, axis=1)
        
        # -- Update Accumulator --
        # Rescale the old accumulator.
        acc = acc * alpha[:, None]
        
        # Add the contribution from the current block: P_ij * V_j
        p_ij = p_ij.to(Q.dtype.element_ty) # Cast P to half for the dot product
        acc += tl.dot(p_ij, v)
        
        # Update the running max and sum for the next iteration.
        m_i = m_i_new
        l_i = l_i_new

    # 6. Final Normalization and Store
    # Normalize the accumulator with the final sum `l_i`.
    # Add a small epsilon to l_i to avoid division by zero.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    # Create pointers to the output matrix O.
    o_ptrs = o_ptr + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    
    # Store the final result.
    tl.store(o_ptrs, acc.to(Q.dtype.element_ty), mask=m_mask[:, None])


def flash_attn_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    PyTorch wrapper for the Triton Flash Attention kernel.
    
    Args:
        q (torch.Tensor): Query tensor of shape (batch, heads, seq_len, head_dim)
        k (torch.Tensor): Key tensor of shape (batch, heads, seq_len, head_dim)
        v (torch.Tensor): Value tensor of shape (batch, heads, seq_len, head_dim)

    Returns:
        torch.Tensor: Output tensor of shape (batch, heads, seq_len, head_dim)
    """
    # 1. Validate inputs
    assert q.shape == k.shape == v.shape
    assert q.dtype == k.dtype == v.dtype
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype in [torch.float16, torch.bfloat16]

    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # 2. Define kernel constants and tuning parameters
    # These values are chosen based on the CUDA kernel's configuration and common practice.
    # Br = 64, Bc = 32 in the CUDA kernel.
    BLOCK_M = 64
    BLOCK_N = 32
    
    # The CUDA kernel supports head_dim 32, 64, 96, 128.
    # Triton can handle this more dynamically.
    assert head_dim in [32, 64, 96, 128], f"Head dimension {head_dim} not supported by this configuration."

    # Create an empty output tensor
    o = torch.empty_like(q)
    
    # 3. Configure and launch the kernel
    scale = 1.0 / math.sqrt(head_dim)
    
    # The grid determines how many instances of the kernel to launch.
    # One instance per row-block of Q, for each batch and head.
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)
    
    # Set number of stages to 2 to match the CUDA kernel's pipelining.
    num_stages = 2
    
    # Set number of warps. 4 is a common, robust choice.
    # The CUDA kernel uses 128 threads = 4 warps.
    num_warps = 4

    _flash_attn_triton_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch_size, num_heads, seq_len,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=head_dim,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    return o

# Example Usage
if __name__ == '__main__':
    # --- Configuration ---
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = 4, 12, 2048, 64
    dtype = torch.float16
    device = 'cuda'

    # --- Create Tensors ---
    q = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), dtype=dtype, device=device, requires_grad=False)
    k = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), dtype=dtype, device=device, requires_grad=False)
    v = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), dtype=dtype, device=device, requires_grad=False)

    # --- Run Triton Kernel ---
    print("Running Triton Flash Attention...")
    triton_output = flash_attn_triton(q, k, v)
    
    # --- Run PyTorch Reference Implementation for Verification ---
    print("Running PyTorch reference implementation...")
    # Note: The reference implementation can be very slow and memory-intensive for large sequences.
    # It's used here to verify correctness.
    scale = 1.0 / math.sqrt(D_HEAD)
    q_scaled = q * scale
    attn_scores = torch.matmul(q_scaled, k.transpose(-2, -1))
    # Masking is not implemented in the Triton kernel for simplicity, but would be added here.
    # For example, for causal attention:
    # mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=device), diagonal=1).bool()
    # attn_scores.masked_fill_(mask, -float('inf'))
    
    attn_probs = torch.softmax(attn_scores, dim=-1)
    pytorch_output = torch.matmul(attn_probs, v)

    # --- Compare Results ---
    print(f"Triton output shape: {triton_output.shape}")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    
    # Use a tolerant comparison due to potential floating point differences
    # between the fused Triton kernel and the step-by-step PyTorch version.
    is_close = torch.allclose(triton_output, pytorch_output, atol=1e-2, rtol=1e-2)
    print(f"Outputs are close: {is_close}")

    # --- Benchmark (Optional) ---
    # ms, min_ms, max_ms = triton.testing.do_bench(lambda: flash_attn_triton(q, k, v))
    # print(f"Triton average time: {ms:.4f} ms")
    
    # PyTorch benchmark can be very slow
    # ms_torch, _, _ = triton.testing.do_bench(lambda: torch.matmul(torch.softmax(torch.matmul(q * scale, k.transpose(-2, -1)), dim=-1), v))
    # print(f"PyTorch average time: {ms_torch:.4f} ms")