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
    # ADDED: Padded block size for heads
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    q_token_ptr = Q_ptr + pid_b * stride_b + pid_s * stride_s
    k_token_ptr = K_ptr + pid_b * stride_b + pid_s * stride_s
    v_token_ptr = V_ptr + pid_b * stride_b + pid_s * stride_s

    # --- CHANGES START HERE ---

    # 1. Create ranges using the padded BLOCK_H size, not NUM_HEADS.
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM)

    # 2. Create a mask to handle the case where NUM_HEADS is not a power of two.
    # This will be True for valid heads and False for padded ones.
    h_mask = offs_h < NUM_HEADS

    # Create pointer blocks as before, but they are now potentially larger.
    q_ptrs = q_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d
    k_ptrs = k_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d
    v_ptrs = v_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d

    # 3. Apply the mask when loading data.
    # `other=0.0` fills the padded rows with zeros, preventing them
    # from affecting the dot product computation.
    q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)
    k = tl.load(k_ptrs, mask=h_mask[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=h_mask[:, None], other=0.0)

    # --- NO CHANGES TO COMPUTATION ---
    # The math works correctly on the zero-padded tensors.
    scores = tl.dot(q, tl.trans(k))
    scaling_factor = (1.0 / tl.sqrt(HEAD_DIM.to(tl.float32)))
    scores *= scaling_factor
    attn_probs = tl.softmax(scores)
    attn_probs = attn_probs.to(tl.float32)
    output_block = tl.dot(attn_probs, v)

    # --- FINAL CHANGE ---
    o_token_ptr = O_ptr + pid_b * stride_b + pid_s * stride_s
    o_ptrs = o_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d
    
    # 4. Apply the mask when storing the result to avoid writing out of bounds.
    tl.store(o_ptrs, output_block, mask=h_mask[:, None])


def mha_triton_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Host wrapper updated to launch the fixed Triton kernel.
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    output = torch.empty_like(q)
    
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    
    grid = (batch_size, seq_len)
    
    # ADDED: Calculate the next power of 2 for num_heads.
    BLOCK_H = triton.next_power_of_2(num_heads)
    
    # Launch the fixed kernel, passing the new BLOCK_H constant.
    mha_triton_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_H=BLOCK_H, # Pass the padded size
    )
    
    return output