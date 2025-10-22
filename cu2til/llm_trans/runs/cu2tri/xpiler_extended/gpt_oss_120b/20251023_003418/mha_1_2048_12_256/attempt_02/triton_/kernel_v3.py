import torch
import triton
import triton.language as tl
import math

# -------------------------------------------------------------------------
# Compile‑time configuration (tuned for the target GPU)
# -------------------------------------------------------------------------
BLOCK_SIZE = 32   # threads per program (must be a multiple of 32)
MAX_HEADS = 32    # upper bound on number of heads (shared‑memory allocation)
CHUNK = 32        # tile size for reduction over head_dim

@triton.jit
def _triton_kernel_impl(
    Q, K, V, output,
    batch_size, seq_len,
    # compile‑time constants
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    # runtime argument
    scaling_factor,
    # compile‑time configuration
    BLOCK_SIZE: tl.constexpr,
    MAX_HEADS: tl.constexpr,
    CHUNK: tl.constexpr
):
    """
    Triton implementation of the original CUDA kernel.
    Tensors are interpreted as (B, S, H, D) in row‑major order.
    """
    pid_batch = tl.program_id(0)   # blockIdx.x → batch index
    pid_seq   = tl.program_id(1)   # blockIdx.y → query index

    lane = tl.arange(0, BLOCK_SIZE)                # threadIdx.x
    head_mask = lane < num_heads                    # mask for valid heads

    # -----------------------------------------------------------------
    # Strides for the (B, S, H, D) layout
    # -----------------------------------------------------------------
    stride_batch = seq_len * num_heads * head_dim   # B‑stride
    stride_seq   = num_heads * head_dim             # S‑stride
    stride_head  = head_dim                         # H‑stride

    # Base offset for the current (batch, seq) pair
    base = pid_batch * stride_batch + * stride_seq

    # -----------------------------------------------------------------
    # 1) Compute Q·Kᵀ → score matrix (num_heads × num_heads)
    # -----------------------------------------------------------------
    # Allocate shared memory for the full score matrix
    score_smem = tl.shared(tl.float32, (MAX_HEADS, MAX_HEADS))

    for n in range(num_heads):               # second head index
        # Accumulator for the dot‑product Q[:,m,:]·K[:,n,:]
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        # Reduce over head_dim in CHUNK‑sized tiles
        for offset in range(0, head_dim, CHUNK):
            dim_offsets = tl.arange(0, CH)
           _mask = (offset + dim_offsets) < head_dim

            # Q slice for head m (shape: BLOCK_SIZE × CHUNK)
_ptr = Q base + lane * stride_head + offset + dim_offsets
            q = tl.load(
                q_ptr,
                mask=head_mask[:, None] & dim_mask[None, :],
                other=0.0
            )

            # K slice for head n (shape: CHUNK)
            k_ptr = K + base + n * stride_head + offset + dim_offsets
            k = tl.load(
                k_ptr,
                mask=dim_mask,
                other=0.0
            )

            # Reduce over CHUNK dimension
            acc += tl.sum(q * k, axis=1)

        # Apply scaling factor 1/√head_dim
        acc = acc * scaling_factor

        # Store raw scores into shared memory (one row per head)
        tl.store(score_smem + (lane, n), acc, mask=head_mask)

    # -----------------------------------------------------------------
    # 2) Softmax across the second head dimension (n)
    # -----------------------------------------------------------------
    exp_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for n in range(num_heads):
        val = tl.load(score_smem + (lane, n), mask=head_mask, other=0.0)
        val = tl.exp(val)
        tl.store(score_smem + (lane, n), val, mask=head_mask)
        exp_sum += val

    # Normalise
    n in range(num_heads):
        val = tl.load(score_smem + (lane, n), mask=head_mask, other=0.0)
        val = val / exp_sum
        tl.store(score_smem + (lane, n), val, mask=head_mask)

    # -----------------------------------------------------------------
    # 3) Weighted sum with V → output
    # -----------------------------------------------------------------
    for d_offset in range(0, head_dim, CHUNK):
        dim_offsets = tl.arange(0, CHUNK)
        dim_mask = (d_offset + dim_offsets) < head_dim

        # Accumulator for the output tile (BLOCK_SIZE × CHUNK)
        out = tl.zeros([BLOCK_SIZE, CHUNK], dtype=tl.float32)

        for k in range(num_heads):
            # V slice for head k (shape: CHUNK)
            v_ptr = V + base + k * stride_head + d_offset + dim_offsets
            v = tl.load(
                v_ptr,
                mask=dim_mask,
                other=0.0
            )

            # Attention weight for (m, k) – one scalar per lane
            s = tl.load(score_smem + (lane, k), mask=head_mask, other=0.0)

            # Broadcast the scalar across CHUNK dimension and accumulate
            out += s[:, None] * v

        # Store the result back to the output tensor
        out_ptr = output + base + lane * stride_head + d_offset + dim_offsets
        tl.store(
            out_ptr,
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
    # -----------------------------------------------------------------
    # Argument validation
    # -----------------------------------------------------------------
    assert queries.is_cuda and keys.is_cuda and.is_cuda and output.is_cuda, \
        "All tensors must be on CUDA"
    assert queries.dtype == torch.float32, "Only float32 tensors are supported"
    assert queries.shape == (batch_size, seq_len, num_heads, head_dim)
    assert keys.shape    == (batch_size, seq_len, num_heads, head_dim)
    assert values.shape  == (batch_size, seq_len, num_heads, head_dim)
    assert output.shape  == (batch_size, seq_len, num_heads, head_dim)

    # Ensure contiguous layout
    queries = queries.contiguous()
    keys    = keys.contiguous()
    values  = values.contiguous()
    output  = output.contiguous()

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
        scaling_factor=scaling_factor,
        num_heads=num_heads,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_HEADS=MAX_HEADS,
        CHUNK=CHUNK,
        num_warps=4,
    )