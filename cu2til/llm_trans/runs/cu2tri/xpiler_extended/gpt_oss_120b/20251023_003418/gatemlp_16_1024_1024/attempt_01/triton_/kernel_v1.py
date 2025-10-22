import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing the CUDA gatemlp_kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    x_ptr, a_ptr, b_ptr, out_ptr,
    batch, K, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Compute:
        acc1 = sum_k x[row, k] * a[k, col]
        acc2 = sum_k x[row, k] * b, col]
        silu = acc1 / (1 + exp(-acc1))
        output[row, col] = silu * acc2
    All accumulations are performed in fp64 for maximal precision.
    """
    pid_m = tl.program_id(0)  # block index over rows (batch dimension)
    pid_n = tl.program_id(1)  # block index over columns (N dimension)

    # Global row/col indices for this program instance
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # shape (BLOCK_M,)
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # shape (BLOCK_N,)

    # Masks for out‑of‑bounds rows/cols
    row_mask = row < batch
    col_mask = col < N
    store_mask = row_mask[:, None] & col_mask[None, :]   # shape (BLOCK_M, BLOCK_N)

    # Double‑precision accumulators
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)

    # ------------------------------------------------------------------
    # Loop over the K dimension in chunks of BLOCK_K
    # ------------------------------------------------------------------
    k = tl.zeros([], dtype=tl.int32)   # scalar start offset = 0
    while k < K:
        # Vector of K indices for this chunk
        k_idx = tl.arange(0, BLOCK_K)          # (BLOCK_K,)
        k_offset = k + k_idx                    # (BLOCK_K,)

        # Valid‑K mask (handles tail of K)
        k_valid = k_offset < K

        # --------------------------------------------------------------
        # Load X block: shape (BLOCK_M, BLOCK_K)
        # --------------------------------------------------------------
        x_offset = row[:, None] * K + k_offset[None, :]   # (BLOCK_M, BLOCK_K)
        x_mask = row_mask[:, None] & k_valid[None, :]
        x = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0).to(tl.float64)

        # --------------------------------------------------------------
        # Load A block: shape (BLOCK_K, BLOCK_N)
        # --------------------------------------------------------------
        a_offset = k_offset[:, None] * N + col[None, :]   # (BLOCK_K, BLOCK_N)
        a_mask = k_valid[:, None] & col_mask[None, :]
        a = tl.load(a_ptr + a_offset, mask=a_mask, other=0.0).to(tl.float64)

        # --------------------------------------------------------------
        # Load B block: shape (BLOCK_K, BLOCK_N)
        # --------------------------------------------------------------
        b_offset = k_offset[:, None] * N + col[None, :]   # (BLOCK_K, BLOCK_N)
        b_mask = k_valid[:, None] & col_mask[None, :]
        b = tl.load(b_ptr + b_offset, mask=b_mask, other=0.0).to(tl.float64)

        # --------------------------------------------------------------
        # Accumulate the dot products for this K‑chunk
        # --------------------------------------------------------------
        acc1 += tl.dot(x, a)   # (BLOCK_M, BLOCK_N)
        acc2 += tl.dot(x, b)

        # Move to the next K‑chunk
        k += BLOCK_K

    # ------------------------------------------------------------------
    # Apply SiLU activation and write the result
    # ------------------------------------------------------------------
    silu = acc1 / (1.0 + tl.exp(-acc1))   # fp64
    out = silu * acc2                     # fp64
    out_f32 = out.to(tl.float32)         # cast to fp32 for storage

    out_offset = row[:, None] * N + col[None, :]   # (BLOCK_M, BLOCK_N)
    tl.store(out_ptr + out_offset, out_f32, mask=store_mask)


# ----------------------------------------------------------------------
# Python wrapper that mimics the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    batch: int,
    K: int,
    N: int,
) -> None:
    """
    Entry point that launches the Triton implementation.
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch, K) with dtype torch.float16.
    a : torch.Tensor
        Weight tensor of shape (K, N) with dtype torch.float16.
    b : torch.Tensor
        Weight tensor of shape (K, N) with dtype torch.float16.
    output : torch.Tensor
        Output tensor of shape (batch, N) with dtype torch.float32.
    batch, K, N : int
        Dimensions matching the shapes above.
    """
    # Basic sanity checks
    assert x.is_cuda and a.is_cuda and b.is_cuda and output.is_cuda, "All tensors must reside on CUDA"
    assert x.dtype == torch.float16 and a.dtype == torch.float16 and b.dtype == torch.float16, "x, a, b must be torch.float16"
    assert output.dtype == torch.float32, "output must be torch.float32"
    assert x.shape == (batch, K), f"x shape mismatch: expected ({batch},{K}), got {x.shape}"
    assert a.shape == (K, N), f"a shape mismatch: expected ({K},{N}), got {a.shape}"
    assert b.shape == (K, N), f"b shape mismatch: expected ({K},{N}), got {b.shape}"
    assert output.shape == (batch, N), f"output shape mismatch: expected ({batch},{N}), got {output.shape}"

    # Ensure contiguous memory layout
    x = x.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    output = output.contiguous()

    # Tunable block sizes – 16×16 work well on Hopper‑class GPUs
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32   # can be increased if K is large and registers permit

    # Grid dimensions (rows over batch, cols over N)
    grid_x = (batch + BLOCK_M - 1) // BLOCK_M
    grid_y = (N + BLOCK_N - 1) // BLOCK_N
    grid = (grid_x, grid_y)

    # Launch the kernel
    _triton_kernel_impl[grid](
        x,
        a,
        b,
        output,
        batch,
        K,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    # Optional: synchronize for timing/debugging
    # torch.cuda.synchronize()