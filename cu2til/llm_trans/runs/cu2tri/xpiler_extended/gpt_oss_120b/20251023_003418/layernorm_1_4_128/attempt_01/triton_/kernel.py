import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Triton kernel implementation
# -------------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr, gamma_ptr, beta_ptr, B_ptr,
    N_ROWS,
    D_MODEL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton implementation of the original CUDA kernel.
    It processes up to 4 rows (idx < 4) exactly as the CUDA version does.
    """
    pid = tl.program_id(0)
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Original CUDA guard: only rows with idx < 4 are processed
    mask_row = (row_idx < N_ROWS) & (row_idx < 4)
    col_idx = tl.arange(0, D_MODEL)

    # Load A (shape: [BLOCK_SIZE, D_MODEL])
    a = tl.load(
        A_ptr + row_idx[:, None] * D_MODEL + col_idx[None, :],
        mask=mask_row[:, None] & (col_idx < D_MODEL),
        other=0.0
    )

    # Compute mean per row
    mean = tl.sum(a, axis=1) / D_MODEL
    # Centered values
    diff = a - mean[:, None]
    # Compute variance (standard deviation)
    var = tl.sqrt(tl.sum(diff * diff, axis=1) / D_MODEL)

    # Load gamma and beta (shape: [D_MODEL])
    gamma = tl.load(gamma_ptr + col_idx, mask=col_idx < D_MODEL, other=0.0)
    beta = tl.load(beta_ptr + col_idx, mask=col_idx < D_MODEL, other=0.0)

    # Broadcast gamma/beta to match rows: [1, D_MODEL] -> [BLOCK_SIZE, D_MODEL]
    gamma = tl.broadcast_to(tl.reshape(gamma, (1, D_MODEL)), (BLOCK_SIZE, D_MODEL))
    beta = tl.broadcast_to(tl.reshape(beta, (1, D_MODEL)), (BLOCK_SIZE, D_MODEL))

    # Normalization: (a - mean) * gamma / (var + eps) + beta
    eps = 1e-5
    norm = diff * gamma / (var[:, None] + eps) + beta

    # Store result into B
    tl.store(
        B_ptr + row_idx[:, None] * D_MODEL + col_idx[None, :],
        norm,
        mask=mask_row[:, None] & (col_idx < D_MODEL)
    )

# -------------------------------------------------------------------------
# Wrapper function (entry point) matching the original CUDA signature
# -------------------------------------------------------------------------
def triton_kernel(A, gamma, beta, B,
                  batch_size, seq_length, d_model):
    """
    Triton wrapper that mimics the original CUDA kernel signature.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (batch_size * seq_length, d_model) **or**
        (batch_size, seq_length, d_model), dtype torch.float32, CUDA device.
    gamma : torch.Tensor
        Scale tensor of shape (d_model,), dtype torch.float32, CUDA device.
    beta : torch.Tensor
        Shift tensor of shape (d_model,), dtype torch.float32, CUDA device.
    B : torch.Tensor
        Output tensor with the same shape as A, dtype torch.float32, CUDA device.
    batch_size : int
        Number of batches.
    seq_length : int
        Sequence length.
    d_model : int
        Model dimension (must match the last dimension of A and B).
    """
    # -----------------------------------------------------------------
    # Sanity checks (mirroring the CUDA wrapper)
    # -----------------------------------------------------------------
    assert A.is_cuda and gamma.is_cuda and beta.is_cuda and B.is_cuda, \
        "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and gamma.dtype == torch.float32 \
           and beta.dtype == torch.float32 and B.dtype == torch.float32, \
        "All tensors must be float32"
    assert gamma.numel() == d_model and beta.numel() == d_model, \
        "gamma and beta must contain d_model elements"

    # Ensure contiguous memory layout (required for pointer arithmetic)
    assert A.is_contiguous() and B.is_contiguous(), \
        "A and B must be contiguous"
    A = A.contiguous()
    B = B.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # -----------------------------------------------------------------
    # Flatten to 2‑D if needed (supports both (N, d_model) and (B, S, d_model))
    # -----------------------------------------------------------------
    A_flat = A.view(-1, d_model)
    B_flat = B.view(-1, d_model)

    # -----------------------------------------------------------------
    # Grid configuration (identical to the CUDA launch)
    # -----------------------------------------------------------------
    block_size = 4                     # matches original blockDim.x
    n_rows = batch_size * seq_length
    num_blocks = (n_rows + block_size - 1) // block_size
    grid = (num_blocks,)

    # -----------------------------------------------------------------
    # Launch the Triton kernel
    # -----------------------------------------------------------------
    _triton_kernel_impl[grid](
        A_flat,
        gamma,
        beta,
        B_flat,
        n_rows,
        D_MODEL=d_model,
        BLOCK_SIZE=block_size,
        num_warps=4,          # reasonable default for this workload
    )
    # Synchronize to guarantee completion before returning
    torch.cuda.synchronize()