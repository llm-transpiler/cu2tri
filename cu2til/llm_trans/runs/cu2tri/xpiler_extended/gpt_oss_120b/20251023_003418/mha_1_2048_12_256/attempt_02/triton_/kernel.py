import torch
import triton
import triton.language as tl
import math

# -------------------------------------------------------------------------
# Compile‑time configuration (tuned for the target GPU)
# -------------------------------------------------------------------------
BLOCK_SIZE = 32   # threads per warp (must be a multiple of 32)
MAX_HEADS = 32    # upper bound on number of heads (kept for compatibility)
CHUNK = 32        # tile size for reduction over head_dim

@triton.jit
def _triton_kernel_impl(
    Q, K, V, output,
    batch_size, seq_len,
    scaling_factor,
    # compile‑time constants
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_HEADS: tl.constexpr,
    CHUNK: tl.constexpr
):
    """
    Triton implementation of the original CUDA kernel.
    Tensors are interpreted as (B, S, H, D) in row‑major order.
    """
    pid_batch = tl.program_id(0)   # blockIdx.x → batch index
    pid_seq = tl.program_id(1)     # blockIdx.y → sequence index

    lane = tl.arange(0, BLOCK_SIZE)          # threadIdx.x (head index)
    head_mask = lane < num_heads               # mask for valid heads
    # replace out‑of‑range lane indices with 0 to avoid OOB address generation
    valid_lane = tl.where(head_mask, lane, 0)

    # -----------------------------------------------------------------
    # Strides for the (B, S, H, D) layout (in elements)
    # -----------------------------------------------------------------
    stride_batch = seq_len * num_heads * head_dim   # B‑stride
    stride_seq = num_heads * head_dim               # S‑stride
    stride_head = head_dim                         # H‑stride

    # Base offset for the current (batch, seq) pair
    base = pid_batch * stride_batch + pid_seq * stride_seq

    # -----------------------------------------------------------------
    # 1) Compute Q·Kᵀ → score matrix (num_heads × num_heads)
    # -----------------------------------------------------------------
    # Accumulate scores for all n in a Python list, then stack.
    score_list = []
    for n in range(num_heads):
        # accumulator for dot‑product Q[m]·K[n] (one scalar per lane)
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for offset in range(0, head_dim, CHUNK):
            dim_offsets = tl.arange(0, CHUNK)
            dim_mask = (offset + dim_offsets) < head_dim

            # Q slice for head m (lane)
            q_ptr = Q + base + valid_lane * stride_head + offset
            q = tl.load(
                q_ptr + dim_offsets,
                mask=head_mask[:, None] & dim_mask[None, :],
                other=0.0
            )

            # K slice for head n
            k_ptr = K + base + n * stride_head + offset
            k = tl.load(
                k_ptr + dim_offsets,
                mask=dim_mask,
                other=0.0
            )

            # Reduce over CHUNK dimension
            acc += tl.sum(q * k, axis=1)   # shape (BLOCK_SIZE,)
        # Apply scaling factor 1/√head_dim
        acc = acc * scaling_factor
        score_list.append(acc)

    # Shape: (BLOCK_SIZE, num_heads)
    scores = tl.stack(score_list, axis=1)

    # -----------------------------------------------------------------
    # 2) Softmax across the second head dimension (n)
    # -----------------------------------------------------------------
    exp_scores = tl.exp(scores)
    sum_exp = tl.sum(exp_scores, axis=1)          # shape (BLOCK_SIZE,)
    softmax = exp_scores / sum_exp[:, None]       # broadcast division

    # -----------------------------------------------------------------
    # 3) Weighted sum with V → output
    # -----------------------------------------------------------------
    for d_offset in range(0, head_dim, CHUNK):
        dim_offsets = tl.arange(0, CHUNK)
        dim_mask = (d_offset + dim_offsets) < head_dim

        # accumulator for the output tile (BLOCK_SIZE, CHUNK)
        out = tl.zeros([BLOCK_SIZE, CHUNK], dtype=tl.float32)

        for k in range(num_heads):
            # V slice for head k
            v_ptr = V + base + k * stride_head + d_offset
            v = tl.load(
                v_ptr + dim_offsets,
                mask=dim_mask,
                other=0.0
            )  # shape (CHUNK,)

            # attention weight for (m, k) – one scalar per lane
            w = softmax[:, k]  # shape (BLOCK_SIZE,)

            # broadcast weight across CHUNK dimension and accumulate
            out += w[:, None] * v[None, :]
        # store result
        out_ptr = output + base + valid_lane * stride_head + d_offset
        tl.store(
            out_ptr + dim_offsets,
            out,
            mask=head_mask[:, None] & dim_mask[None, :]
        )

def triton_kernel(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int
):
    """
    Wrapper that mirrors the original ``cuda_kernel`` signature.
    All tensors must be contiguous, reside on the same CUDA device,
    and have dtype torch.float32 with shape
    (batch_size, seq_len, num_heads, head_dim).
    """
    assert queries.is_cuda and keys.is_cuda and values.is_cuda and output.is_cuda, \
        "All tensors must be on CUDA"
    assert queries.dtype == torch.float32, "Only float32 tensors are supported"
    assert queries.shape == (batch_size, seq_len, num_heads, head_dim)
    assert keys.shape == (batch_size, seq_len, num_heads, head_dim)
    assert values.shape == (batch_size, seq_len, num_heads, head_dim)
    assert output.shape == (batch_size, seq_len, num_heads, head_dim)

    # Ensure contiguous layout
    queries = queries.contiguous()
    keys = keys.contiguous()
    values = values.contiguous()
    output = output.contiguous()

    scaling_factor = 1.0 / math.sqrt(head_dim)

    # Grid: one program per (batch, sequence) pair
    grid = (batch_size, seq_len)

    _triton_kernel_impl[grid](
        queries,
        keys,
        values,
        output,
        batch_size,
        seq_len,
        scaling_factor,
        num_heads=num_heads,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_HEADS=MAX_HEADS,
        CHUNK=CHUNK,
        num_warps=1,
    )