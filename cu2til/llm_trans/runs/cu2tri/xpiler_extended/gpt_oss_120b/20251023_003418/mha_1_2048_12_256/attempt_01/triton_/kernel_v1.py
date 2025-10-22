import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(Q, K, V, output,
                        batch_size, seq_len, head_dim,
                        NUM_HEADS: tl.constexpr,
                        BLOCK_HEAD_DIM: tl.constexpr):
    # Program IDs correspond to (batch, sequence position, head)
    b = tl.program_id(0)  # batch index
    s = tl.program_id(1)  # sequence position
    h = tl.program_id(2)  # head index (0 <= h < NUM_HEADS)

    # Range over the head dimension (the inner‑most dimension)
    dim_range = tl.arange(0, BLOCK_HEAD_DIM)

    # Strides for the 4‑D layout (batch, seq_len, head, dim)
    stride_batch = seq_len * NUM_HEADS * BLOCK_HEAD_DIM
    stride_seq   = NUM_HEADS * BLOCK_HEAD_DIM
    stride_head  = BLOCK_HEAD_DIM

    # -------------------------------------------------------------------------
    # Load Q for the current head
    # -------------------------------------------------------------------------
    q_offset = b * stride_batch + s * stride_seq + h * stride_head + dim_range
    q = tl.load(Q + q_offset)

    # -------------------------------------------------------------------------
    # Compute attention scores = Q·Kᵀ for all heads (NUM_HEADS)
    # -------------------------------------------------------------------------
    # Shared memory to hold the score row for this head
    score_smem = tl.shared([NUM_HEADS], dtype=tl.float32)

    scaling = 1.0 / tl.sqrt(tl.float32(BLOCK_HEAD_DIM))

    for n in range(NUM_HEADS):
        # Load K for head n
        k_offset = b * stride_batch + s * stride_seq + n * stride_head + dim_range
        k = tl.load(K + k_offset)
        # Dot product Q·K_n
        dot = tl.dot(q, k)
        # Store scaled score into shared memory
        tl.store(score_smem + n, dot * scaling)

    # -------------------------------------------------------------------------
    # Softmax over the NUM_HEADS dimension
    # -------------------------------------------------------------------------
    head_range = tl.arange(0, NUM_HEADS)
    scores = tl.load(score_smem + head_range)          # shape (NUM_HEADS,)
    scores_exp = tl.exp(scores)
    sum_exp = tl.sum(scores_exp, axis=0)               # scalar
    probs = scores_exp / sum_exp                        # softmax probabilities

    # -------------------------------------------------------------------------
    # Weighted sum of V using the softmax probabilities
    # -------------------------------------------------------------------------
    out = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)

    for n in range(NUM_HEADS):
        v_offset = b * stride_batch + s * stride_seq + n * stride_head + dim_range
        v = tl.load(V + v_offset)
        out += probs[n] * v

    # -------------------------------------------------------------------------
    # Write the result
    # -------------------------------------------------------------------------
    out_offset = b * stride_batch + s * stride_seq + h * stride_head + dim_range
    tl.store(output + out_offset, out)


def triton_kernel(queries, keys, values, output,
                  batch_size, seq_len, num_heads, head_dim):
    """
    Triton implementation of the original CUDA attention kernel.
    Parameters
    ----------
    queries, keys, values, output : torch.Tensor
        Tensors of shape (batch_size, seq_len, num_heads, head_dim) and dtype torch.float32.
    batch_size, seq_len, num_heads, head_dim : int
        Dimensions of the tensors    """
    # Ensure tensors are contiguous and reside on the same CUDA device
    queries = queries.contiguous()
    keys   = keys.contiguous()
    values = values.contiguous()
    output = output.contiguous()

    # Grid configuration matches the original CUDA launch: (batch, seq_len, num_heads)
    grid = (batch_size, seq_len, num_heads)

    # Launch the Triton kernel. NUM_HEADS and BLOCK_HEAD_DIM are compile‑time constants.
    _triton_kernel_impl[grid](
        queries,
        keys,
        values,
        output,
        batch_size,
        seq_len,
        head_dim,
        NUM_HEADS=num_heads,
        BLOCK_HEAD_DIM=head_dim,
        num_warps=4,          # reasonable default for this workload
    )