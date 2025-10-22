import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Triton kernel implementation
# -------------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr, gamma_ptr, beta_ptr, B_ptr,
    N_ROWS,                     # total number of rows (batch_size * seq_length)
    D_MODEL: tl.constexpr,      # model dimension (e.g., 128)
    BLOCK_SIZE: tl.constexpr    # number of rows processed per program (fixed to 4)
):
    """
    Triton implementation of the original CUDA kernel.
    It processes up to 4 rows (idx < 4) exactly as the CUDA version does.
    """
    # -----------------------------------------------------------------
    # 1) Compute row indices for this program
    # -----------------------------------------------------------------
    pid = tl.program_id(0)                     # program (block) index
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # shape [BLOCK_SIZE]

    # Mask for valid rows: must be within N_ROWS and also < 4 (original CUDA guard)
    mask_row = (row_idx < N_ROWS) & (row_idx < 4)   # shape [BLOCK_SIZE]

    # -----------------------------------------------------------------
    # 2) Column indices
    # -----------------------------------------------------------------
    col_idx = tl.arange(0, D_MODEL)               # shape [D_MODEL]

    # -----------------------------------------------------------------
    # 3) Load A (shape: [BLOCK_SIZE, D_MODEL])
    # -----------------------------------------------------------------
    # Pointer arithmetic: A[row, col] = A_ptr + row * D_MODEL + col
    a = tl.load(
        A_ptr + row_idx[:, None] * D_MODEL + col_idx[None, :],
        mask=mask_row[:, None] & (col_idx < D_MODEL),
        other=0.0
    )  # a: [BLOCK_SIZE, D_MODEL]

    # -----------------------------------------------------------------
    # 4) Compute mean per row
    # -----------------------------------------------------------------
    mean = tl.sum(a, axis=1) / D_MODEL          # shape [BLOCK_SIZE]

    # -----------------------------------------------------------------
    # 5) Centered values (diff = a - mean)
    # -----------------------------------------------------------------
    diff = a - mean[:, None]                     # shape [BLOCK_SIZE, D_MODEL]

    # -----------------------------------------------------------------
    # 6) Compute variance (standard deviation)
    # -----------------------------------------------------------------
    var = tl.sqrt(tl.sum(diff * diff, axis=1) / D_MODEL)   # shape [BLOCK_SIZE]

    # -----------------------------------------------------------------
    # 7) Load gamma and beta (shape: [D_MODEL])
    # -----------------------------------------------------------------
    gamma = tl.load(gamma_ptr + col_idx, mask=col_idx < D_MODEL, other=0.0)
    beta  = tl.load(beta_ptr  + col_idx, mask=col_idx < D_MODEL, other=0.0)

    # Broadcast gamma/beta to match rows: from [D_MODEL] -> [1, D_MODEL] -> [BLOCK_SIZE, D_MODEL]
    gamma = tl.broadcast_to(tl.reshape(gamma, (1, D_MODEL)), (BLOCK_SIZE, D_MODEL))
    beta  = tl.broadcast_to(tl.reshape(beta,  (1, D_MODEL)), (BLOCK_SIZE, D_MODEL))

    # -----------------------------------------------------------------
    # 8) Normalization: (a - mean) * gamma / (var + eps) + beta
    # -----------------------------------------------------------------
    eps = tl.float32(1e-5)
    norm = diff * gamma / (var[:, None] + eps) + beta   # shape [BLOCK_SIZE, D_MODEL]

    # -----------------------------------------------------------------
    # 9) Store result into B
    # -----------------------------------------------------------------
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
        Input tensor of shape (batch_size * seq_length, d_model), dtype torch.float32.
    gamma : torch.Tensor
        Scale tensor of shape (d_model,), dtype torch.float32.
    beta : torch.Tensor
        Shift tensor of shape (d_model,), dtype torch.float32.
    B : torch.Tensor
        Output tensor of the same shape as A, dtype torch.float32.
    batch_size : int
        Number of batches.
    seq_length : int
        Sequence length.
    d_model : int
        Model dimension (must equal A.shape[1] and B.shape[1]).
    """
    # -----------------------------------------------------------------
    # Basic sanity checks (mirroring the CUDA wrapper)
    # -----------------------------------------------------------------
    assert A.is_cuda and gamma.is_cuda and beta.is_cuda and B.is_cuda, \
        "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and gamma.dtype == torch.float32 \
           and beta.dtype == torch.float32 and B.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.shape == (batch_size * seq_length, d_model), \
        f"A shape mismatch: expected {(batch_size * seq_length, d_model)}, got {A.shape}"
    assert B.shape == (batch_size * seq_length, d_model), \
        f"B shape mismatch: expected {(batch_size * seq_length, d_model)}, got {B.shape}"
    assert gamma.shape == (d_model,) and beta.shape == (d_model,), \
        "gamma and beta must be 1-D tensors of length d_model"

    # Ensure contiguous memory layout (required for pointer arithmetic)
    A = A.contiguous()
    B = B.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # -----------------------------------------------------------------
    # Grid configuration (identical to the CUDA launch)
    # -----------------------------------------------------------------
    block_size = 4  # matches the original CUDA blockDim.x
    n_rows = batch_size * seq_length
    num_blocks = (n_rows + block_size - 1) // block_size
    grid = (num_blocks,)

    # -----------------------------------------------------------------
    # Launch the Triton kernel
    # -----------------------------------------------------------------
    _triton_kernel_impl[grid](
        A,
        gamma,
        beta,
        B,
        n_rows,
        D_MODEL=d_model,
        BLOCK_SIZE=block_size,
        num_warps=4,          # reasonable default for this workload
    )
    # Synchronize to make sure the kernel has finished before returning
    torch.cuda.synchronize()