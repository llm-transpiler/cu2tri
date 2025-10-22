import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(Q, K, V, output,
                        batch_size, seq_len, head_dim,
                        NUM_HEADS: tl.constexpr,
                        BLOCK_HEAD_DIM: tl.constexpr,
                        MAX_HEADS: tl.constexpr):
    # Program IDs: (batch, sequence position, head)
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    # Range over the inner dimension (head_dim)
    dim_range = tl.arange(0, BLOCK_HEAD_DIM)

    # Strides for layout (B, S, H, D)
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
    scaling = 1.0 / (BLOCK_HEAD_DIM ** 0.5)  # 1/√head_dim
    scores = tl.zeros([MAX_HEADS], dtype=tl.float32)  # padded to power‑of‑2 size

    for n in range(NUM_HEADS):
        k_offset = b * stride_batch + s * stride_seq + n * stride_head + dim_range
        k = tl.load(K + k_offset)
        dot = tl.dot(q, k)                     # scalar dot product
        scores[n] = dot * scaling               # store scaled score

    # -------------------------------------------------------------------------
    # Softmax over the NUM_HEADS dimension (manual implementation)
    # -------------------------------------------------------------------------
    for n in range(NUM_HEADS):
        scores[n] = tl.exp(scores[n])

    sum_exp = tl.zeros([1], dtype=tl.float32)[0]   # scalar accumulator
    for n in range(NUM_HEADS):
        sum_exp += scores[n]

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
    Wrapper that launches the Triton kernel with the same signature as the original CUDA kernel.
    """
    # Ensure tensors are contiguous and on the same CUDA device
    queries = queries.contiguous()
    keys   = keys.contiguous()
    values = values.contiguous()
    output = output.contiguous()

    # Pad number of heads to the next power of two for Triton's shape constraint
    max_heads = 1 << ((num_heads - 1).bit_length())

    # Grid matches the original CUDA launch: (batch, seq_len, num_heads)
    grid = (batch_size, seq_len, num_heads)

    # Launch the kernel
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