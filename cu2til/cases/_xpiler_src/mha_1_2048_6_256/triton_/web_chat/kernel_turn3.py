import torch
import triton
import triton.language as tl

@triton.jit
def mha_triton_kernel(
    # Pointers to tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride variables
    stride_b, stride_s, stride_h, stride_d,
    # Constants
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    q_token_ptr = Q_ptr + pid_b * stride_b + pid_s * stride_s
    k_token_ptr = K_ptr + pid_b * stride_b + pid_s * stride_s
    v_token_ptr = V_ptr + pid_b * stride_b + pid_s * stride_s

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM)
    h_mask = offs_h < NUM_HEADS

    q_ptrs = q_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d
    k_ptrs = k_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d
    v_ptrs = v_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d

    q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)
    k = tl.load(k_ptrs, mask=h_mask[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=h_mask[:, None], other=0.0)

    # --- CHANGES START HERE: Replace tl.dot with manual matmul ---

    # Original failing line: scores = tl.dot(q, tl.trans(k))
    # Manual implementation for Q @ K.T
    # q is (BLOCK_H, HEAD_DIM), k is (BLOCK_H, HEAD_DIM)
    q_exp = tl.expand_dims(q, 1)      # Shape: (BLOCK_H, 1, HEAD_DIM)
    k_exp = tl.expand_dims(k, 0)      # Shape: (1, BLOCK_H, HEAD_DIM)
    # Element-wise product -> sum over the K dimension (HEAD_DIM)
    scores = tl.sum(q_exp * k_exp, axis=2) # Shape: (BLOCK_H, BLOCK_H)

    # --- Softmax computation remains the same ---
    scaling_factor = (1.0 / tl.sqrt(HEAD_DIM.to(tl.float32)))
    # Apply mask to scores before softmax to zero out padded rows/cols
    scores = tl.where(h_mask[:, None] & h_mask[None, :], scores, float("-inf"))
    scores *= scaling_factor
    attn_probs = tl.softmax(scores)
    attn_probs = attn_probs.to(tl.float32)

    # Original failing line: output_block = tl.dot(attn_probs, v)
    # Manual implementation for Scores @ V
    # attn_probs is (BLOCK_H, BLOCK_H), v is (BLOCK_H, HEAD_DIM)
    p_exp = tl.expand_dims(attn_probs, 2)  # Shape: (BLOCK_H, BLOCK_H, 1)
    v_exp = tl.expand_dims(v, 0)           # Shape: (1, BLOCK_H, HEAD_DIM)
    # Element-wise product -> sum over the common dimension (BLOCK_H)
    output_block = tl.sum(p_exp * v_exp, axis=1) # Shape: (BLOCK_H, HEAD_DIM)

    # --- Storing the result remains the same ---
    o_token_ptr = O_ptr + pid_b * stride_b + pid_s * stride_s
    o_ptrs = o_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d
    tl.store(o_ptrs, output_block, mask=h_mask[:, None])

# The Python wrapper remains the same as the previous fix
def mha_triton_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, num_heads, head_dim = q.shape
    output = torch.empty_like(q)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    grid = (batch_size, seq_len)
    BLOCK_H = triton.next_power_of_2(num_heads)

    mha_triton_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_H=BLOCK_H,
    )
    return output