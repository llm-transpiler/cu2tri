import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(Q, K, V, output,
                        batch_size, seq_len, head_dim,
                        NUM_HEADS: tl.constexpr,
                        BLOCK_HEAD_DIM: tl.constexpr,
                        MAX_HEADS: tl.constexpr):
    """
    Triton implementation of the original CUDA attention kernel.
    Each program instance computes the output for a single (batch, seq, head) tuple.
    """
    # -------------------------------------------------------------------------
    # Program IDs
    # -------------------------------------------------------------------------
    b = tl.program_id(0)  # batch index
    s = tl.program_id(1)  # sequence position
    h = tl.program_id(2)  # head index (0 <= h < NUM_HEADS)

    # -------------------------------------------------------------------------
    # Index range for the innermost dimension (head_dim)
    # -------------------------------------------------------------------------
    dim_range = tl.arange(0, BLOCK_HEAD_DIM)

    # -------------------------------------------------------------------------
    # Strides for the 4‑D layout (B, S, H, D)
    # -------------------------------------------------------------------------
    stride_batch = seq_len * NUM_HEADS * BLOCK_HEAD_DIM
    stride_seq   = NUM_HEADS * BLOCK_HEAD_DIM
    stride_head  = BLOCK_HEAD_DIM

    # -------------------------------------------------------------------------
    # Load Q for the current head
    # -------------------------------------------------------------------------
    q_offset = b * stride_batch + s * stride_seq + h * stride_head + dim_range
    q = tl.load(Q + q_offset)

    # -------------------------------------------------------------------------
    # Compute scaled dot‑product scores Q·Kᵀ for all heads
    # -------------------------------------------------------------------------
    scaling = 1.0 / tl.sqrt(tl.float32(BLOCK_HEAD_DIM))
    # Allocate a padded buffer for scores (required to be power‑of‑2 sized)
    scores = tl.zeros([MAX_HEADS], dtype=tl.float32)

    for n in range(NUM_HEADS):
        k_offset = b * stride_batch + s * stride_seq + n * stride_head + dim_range
        k = tl.load(K + k_offset)
        # 1‑D dot product via elementwise multiply + reduction
        dot = tl.sum(q * k)
        scores[n] = dot * scaling

    # -------------------------------------------------------------------------
    # Softmax over the NUM_HEADS dimension
    # -------------------------------------------------------------------------
    # Exponentiate the valid scores
    for n in range(NUM_HEADS):
        scores[n] = tl.exp(scores[n])

    # Compute sum of exponentials (scalar)
    sum_exp = tl.float32(0.0)
    for n in range(NUM_HEADS):
        sum_exp += scores[n]

    # Normalized probabilities (padded buffer)
    probs = tl.zeros([MAX_HEADS], dtype=tl.float32)
    for n in range(NUM_HEADS):
        probs[n] = scores[n] / sum_exp

    # -------------------------------------------------------------------------
    # Weighted sum of V using the softmax probabilities
    # -------------------------------------------------------------------------
    out = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)

    for n in range(NUM_HEADS):
        v_offset = b * stride_batch + s * stride_seq + n * stride_head + dim_range
        v = tl.load(V + v_offset)
        out += probs[n] * v

    # -------------------------------------------------------------------------
    # Write the result for the current head
    # -------------------------------------------------------------------------
    out_offset = b * stride_batch + s * stride_seq + h * stride_head + dim_range
    tl.store(output + out_offset, out)


def triton_kernel(queries, keys, values, output,
                  batch_size, seq_len, num_heads, head_dim):
    """
    Wrapper that launches the Triton kernel with the exact signature of the
    original CUDA kernel.
    """
    # Ensure tensors are contiguous and reside on the same CUDA device
    queries = queries.contiguous()
    keys   = keys.contiguous()
    values = values.contiguous()
    output = output.contiguous()

    # Pad number of heads to the next power of two (required by tl.zeros)
    max_heads = 1 << ((num_heads - 1).bit_length())

    # Grid configuration matches the original CUDA launch: (batch, seq_len, num_heads)
    grid = (batch_size, seq_len, num_heads)

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
        MAX_HEADS=max_heads,
        num_warps=4,          # reasonable default for this workload
    )