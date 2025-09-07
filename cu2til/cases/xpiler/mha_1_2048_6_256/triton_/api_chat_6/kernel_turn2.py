import torch
import triton
import triton.language as tl

@triton.jit
def attention_kernel_triton(
    # Pointers to tensors
    Q_ptr, K_ptr, V_ptr, output_ptr,
    # Stride variables for memory access
    stride_q_bs, stride_q_sl, stride_q_h, stride_q_d,
    stride_k_bs, stride_k_sl, stride_k_h, stride_k_d,
    stride_v_bs, stride_v_sl, stride_v_h, stride_v_d,
    stride_o_bs, stride_o_sl, stride_o_h, stride_o_d,
    # Matrix dimensions
    num_heads,
    head_dim,
    # Triton block sizes (powers of 2 for performance)
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Triton kernel for intra-token head attention.
    Each program instance computes the attention for a single token across all its heads.
    """
    # 1. Get program IDs to identify the current token
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    # 2. Compute pointers to the start of the Q, K, V slices for the current token
    q_start_ptr = Q_ptr + batch_idx * stride_q_bs + seq_idx * stride_q_sl
    k_start_ptr = K_ptr + batch_idx * stride_k_bs + seq_idx * stride_k_sl
    v_start_ptr = V_ptr + batch_idx * stride_v_bs + seq_idx * stride_v_sl

    # 3. Create ranges and masks for loading data blocks
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_ptrs = q_start_ptr + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d
    # --- CHANGE: Load K directly, not transposed ---
    k_ptrs = k_start_ptr + offs_h[:, None] * stride_k_h + offs_d[None, :] * stride_k_d
    v_ptrs = v_start_ptr + offs_h[:, None] * stride_v_h + offs_d[None, :] * stride_v_d

    mask_h = offs_h < num_heads
    mask_d = offs_d < head_dim

    # 4. Load Q, K, and V slices from HBM into SRAM
    q = tl.load(q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
    # --- CHANGE: Load K instead of K.T ---
    k = tl.load(k_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
    v = tl.load(v_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)

    # --- CHANGE: Replace tl.dot with manual matmul for S = Q @ K.T ---
    # 5. Compute S = Q @ K.T
    # Original tl.dot is not used due to small dimension size (num_heads < 16)
    # We implement it manually with broadcasting.
    # q: [BLOCK_H, BLOCK_D] -> [BLOCK_H, 1, BLOCK_D]
    # k: [BLOCK_H, BLOCK_D] -> [1, BLOCK_H, BLOCK_D]
    # result of product: [BLOCK_H, BLOCK_H, BLOCK_D]
    # sum over axis=2: [BLOCK_H, BLOCK_H]
    q_expanded = tl.expand_dims(q, 1)
    k_expanded = tl.expand_dims(k, 0)
    scores = tl.sum(q_expanded * k_expanded, axis=2)
    # --- END CHANGE ---

    # 6. Apply scaling and softmax
    scaling_factor = (1.0 / tl.sqrt(head_dim.to(tl.float32)))
    scores *= scaling_factor

    score_mask = mask_h[:, None] & mask_h[None, :]
    scores = tl.where(score_mask, scores, -float('inf'))

    row_max = tl.max(scores, axis=1)
    scores -= row_max[:, None]
    numerator = tl.exp(scores)
    denominator = tl.sum(numerator, axis=1)
    softmax_scores = numerator / denominator[:, None]
    
    softmax_scores = tl.where(score_mask, softmax_scores, 0.0).to(v.dtype)

    # --- CHANGE: Replace tl.dot with manual matmul for O = P @ V ---
    # 7. Compute final output O = P @ V
    # softmax_scores (P): [BLOCK_H, BLOCK_H] -> [BLOCK_H, BLOCK_H, 1]
    # v (V):              [BLOCK_H, BLOCK_D] -> [1, BLOCK_H, BLOCK_D]
    # result of product:  [BLOCK_H, BLOCK_H, BLOCK_D]
    # sum over axis=1:    [BLOCK_H, BLOCK_D]
    p_expanded = tl.expand_dims(softmax_scores, 2)
    v_expanded = tl.expand_dims(v, 0)
    output = tl.sum(p_expanded * v_expanded, axis=1)
    # --- END CHANGE ---

    # 8. Write the output block back to HBM
    output_start_ptr = output_ptr + batch_idx * stride_o_bs + seq_idx * stride_o_sl
    output_ptrs = output_start_ptr + offs_h[:, None] * stride_o_h + offs_d[None, :] * stride_o_d
    output_mask = mask_h[:, None] & mask_d[None, :]
    tl.store(output_ptrs, output, mask=output_mask)


# The launcher function remains exactly the same
def triton_kernel_launcher(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    Python launcher for the Triton attention kernel.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    output = torch.empty_like(q)
    
    grid = (batch_size, seq_len)

    BLOCK_H = triton.next_power_of_2(num_heads)
    BLOCK_D = triton.next_power_of_2(head_dim)

    attention_kernel_triton[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        num_heads,
        head_dim,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
    )
    
    return output

# --- Example Usage and Verification ---
if __name__ == '__main__':
    batch_size = 1
    seq_len = 2048
    num_heads = 6
    head_dim = 256

    torch.manual_seed(0)
    q = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda', dtype=torch.float32)
    k = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda', dtype=torch.float32)
    v = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda', dtype=torch.float32)

    triton_output = triton_kernel_launcher(q, k, v)

    import math
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    p = torch.nn.functional.softmax(scores, dim=-1)
    pytorch_output = torch.matmul(p, v)
    
    print("Triton vs. PyTorch allclose:", torch.allclose(triton_output, pytorch_output, atol=1e-5, rtol=1e-4))
    
    print("\nTriton output (slice):")
    print(triton_output[0, 0, :3, :5])
    
    print("\nPyTorch output (slice):")
    print(pytorch_output[0, 0, :3, :5])
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/xpiler/mha_1_2048_6_256/triton_/api_chat_6/kernel_turn2.py
Triton vs. PyTorch allclose: True

Triton output (slice):
tensor([[-0.1168, -0.5989, -0.4364,  0.4442, -0.1245],
        [-1.1084, -0.2109, -0.6088,  0.8212, -0.5519],
        [-0.2770, -0.0590, -0.6516,  0.4019, -0.2279]], device='cuda:0')

PyTorch output (slice):
tensor([[-0.1168, -0.5989, -0.4364,  0.4442, -0.1245],
        [-1.1084, -0.2109, -0.6088,  0.8212, -0.5519],
        [-0.2770, -0.0590, -0.6516,  0.4019, -0.2279]], device='cuda:0')
'''