import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, gamma_ptr, beta_ptr, B_ptr,
                        N_ROWS,
                        D_MODEL: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr):
    # Program ID and row indices
    pid = tl.program_id(0)
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Original CUDA kernel only processes rows with idx < 4
    mask_row = (row_idx < N_ROWS) & (row_idx < 4)

    # Column indices
    col_idx = tl.arange(0, D_MODEL)

    # Load A (shape: [BLOCK_SIZE, D_MODEL])
    a = tl.load(A_ptr + row_idx[:, None] * D_MODEL + col_idx[None, :],
                mask=mask_row[:, None],
                other=0.0)

    # Compute mean per row
    mean = tl.sum(a, axis=1) / D_MODEL

    # Centered values
    diff = a - mean[:, None]

    # Compute variance (standard deviation)
    var = tl.sqrt(tl.sum(diff * diff, axis=1) / D_MODEL)

    # Load gamma and beta (shape: [D_MODEL])
    gamma = tl.load(gamma_ptr + col_idx, mask=col_idx < D_MODEL, other=0.0)
    beta = tl.load(beta_ptr + col_idx, mask=col_idx < D_MODEL, other=0.0)

    # Broadcast to match rows
    gamma = tl.broadcast_to(gamma, (BLOCK_SIZE, D_MODEL))
    beta = tl.broadcast_to(beta, (BLOCK_SIZE, D_MODEL))

    # Normalization: (a - mean) * gamma / (var + eps) + beta
    eps = tl.float32(1e-5)
    norm = diff * gamma / (var[:, None] + eps) + beta

    # Store result back to B
    tl.store(B_ptr + row_idx[:, None] * D_MODEL + col_idx[None, :],
             norm,
             mask=mask_row[:, None])

def triton_kernel(A, gamma, beta, B, batch_size, seq_length, d_model):
    """
    Triton implementation of the original CUDA kernel.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (batch_size * seq_length, d_model), dtype torch.float32.
    gamma : torch.Tensor
        Scale tensor of shape (d_model,), dtype torch.float32.
    beta : torch.Tensor
        Shift tensor of shape (d_model,), dtype torch.float32.
    B : torch.Tensor
        Output tensor of same shape as A, dtype torch.float32.
    batch_size : int
        Batch dimension.
    seq_length : int
        Sequence length dimension.
    d_model : int
        Model dimension (must match the last dimension of A and B).
    """
    # Basic sanity checks
    assert A.is_cuda and gamma.is_cuda and beta.is_cuda and B.is_cuda, "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and gamma.dtype == torch.float32 and beta.dtype == torch.float32 and B.dtype == torch.float32, "All tensors must be float32"
    # Ensure contiguous memory layout
    A = A.contiguous()
    B = B.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # Total number of rows (batch * seq)
    n_rows = batch_size * seq_length

    block_size = 4
    num_blocks = (n_rows + block_size - 1) // block_size
    grid = (num_blocks,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A,
        gamma,
        beta,
        B,
        n_rows,
        D_MODEL=d_model,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    # Synchronize to guarantee completion before returning
    torch.cuda.synchronize()