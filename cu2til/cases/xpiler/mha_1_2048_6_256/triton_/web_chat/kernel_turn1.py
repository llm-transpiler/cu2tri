import torch
import triton
import triton.language as tl

@triton.jit
def mha_triton_kernel(
    # Pointers to tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride variables for tensor memory layout
    stride_b, stride_s, stride_h, stride_d,
    # Compile-time constants for performance
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Triton kernel for a non-standard Multi-Head Attention operation.
    Computes attention between heads for the same token.
    """
    # === 1. Get Program Identifiers ===
    # Each program instance computes the output for a single token in the sequence.
    pid_b = tl.program_id(0)  # Current batch index
    pid_s = tl.program_id(1)  # Current sequence index

    # === 2. Load Data Blocks ===
    # Calculate base pointers for the current token (batch, seq_pos).
    # All subsequent operations are relative to this token.
    q_token_ptr = Q_ptr + pid_b * stride_b + pid_s * stride_s
    k_token_ptr = K_ptr + pid_b * stride_b + pid_s * stride_s
    v_token_ptr = V_ptr + pid_b * stride_b + pid_s * stride_s

    # Create ranges to load all heads and the full head dimension.
    # Note: arange range must be a power of 2, so we use the next power of 2
    NEXT_POWER_OF_2_HEADS = tl.next_power_of_2(NUM_HEADS)
    NEXT_POWER_OF_2_HEAD_DIM = tl.next_power_of_2(HEAD_DIM)
    
    offs_h = tl.arange(0, NEXT_POWER_OF_2_HEADS)    # Head offsets with power-of-2 size
    offs_d = tl.arange(0, NEXT_POWER_OF_2_HEAD_DIM) # Dimension offsets with power-of-2 size
    
    # Mask to only access valid indices
    head_mask = offs_h < NUM_HEADS
    dim_mask = offs_d < HEAD_DIM

    # Create 2D blocks of pointers to load the Q, K, and V data for all heads
    # of the current token into SRAM.
    q_ptrs = q_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d
    k_ptrs = k_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d
    v_ptrs = v_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d
    
    # Load the [NUM_HEADS, HEAD_DIM] blocks.
    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)

    # === 3. Compute Scores (Q @ K.T) & Softmax ===
    # Calculate dot product between heads: Q[h, d] @ K.T[d, h]
    # The result 'scores' is a [NUM_HEADS, NUM_HEADS] block.
    scores = tl.dot(q, tl.trans(k))

    # Apply the scaling factor.
    scaling_factor = (1.0 / tl.sqrt(HEAD_DIM.to(tl.float32)))
    scores *= scaling_factor
    
    # Apply softmax row-wise (along the last axis of the 'scores' block).
    attn_probs = tl.softmax(scores)
    
    # Ensure the probabilities are in float32 for the next matmul.
    attn_probs = attn_probs.to(tl.float32)

    # === 4. Compute Final Output (Scores @ V) ===
    # Multiply the attention probabilities with the V block.
    # attn_probs[h, h] @ V[h, d] -> output_block[h, d]
    output_block = tl.dot(attn_probs, v)

    # === 5. Store Result ===
    # Calculate output pointers and store the resulting block.
    o_token_ptr = O_ptr + pid_b * stride_b + pid_s * stride_s
    o_ptrs = o_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d
    tl.store(o_ptrs, output_block)


def mha_triton_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Host wrapper function to launch the Triton kernel for MHA.
    """
    # Get tensor dimensions: (batch, seq_len, num_heads, head_dim)
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Pre-allocate the output tensor on the same device as input.
    output = torch.empty_like(q)
    
    # Ensure tensors are contiguous in memory for predictable stride calculations.
    # This is a common requirement for custom kernels.
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    
    # The grid defines how many instances of the kernel to launch.
    # We launch one instance for each token in the batch.
    grid = (batch_size, seq_len)
    
    # Launch the kernel.
    mha_triton_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
    )
    
    return output