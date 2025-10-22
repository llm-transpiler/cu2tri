import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel (named exactly as required)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A,          # *float32, shape [total_len, FEATURE_SIZE]
    gamma,      # *float32, shape [FEATURE_SIZE]
    beta,       # *float32, shape [FEATURE_SIZE]
    B,          # *float32, shape [total_len, FEATURE_SIZE]
    total_len,  # int32, total number of rows = batch_size * seq_length
    BLOCK_SIZE: tl.constexpr,   # compile‑time constant: threads per program (matches CUDA blockDim.x)
    FEATURE_SIZE: tl.constexpr, # compile‑time constant: feature dimension (hard‑coded to 128)
):
    """
    Triton implementation of the original CUDA kernel.
    Each program processes BLOCK_SIZE rows; the original kernel only
    operates on rows with idx < 4, which we preserve via a mask.
    """
    # ------------------------------------------------------------------
    # 1) Compute logical row indices for this program.
    # ------------------------------------------------------------------
    row_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask for rows that belong to the workload and respect the original idx < 4 condition.
    valid = (row_idx < total_len) & (row_idx < 4)

    # ------------------------------------------------------------------
    # 2) Load the 128‑element vectors for the selected rows.
    # ------------------------------------------------------------------
    col_offset = tl.arange(0, FEATURE_SIZE)                     # [0, 1, ..., FEATURE_SIZE‑1]
    offsets = row_idx[:, None] * FEATURE_SIZE + col_offset[None, :]  # (BLOCK_SIZE, FEATURE_SIZE)

    # Load A; out‑of‑range elements are filled with 0.0 (masked later)
    a = tl.load(A + offsets, mask=valid[:, None], other=0.0)

    # ------------------------------------------------------------------
    # 3) Compute mean per row.
    # ------------------------------------------------------------------
    mean = tl.sum(a, axis=1) / FEATURE_SIZE

    # ------------------------------------------------------------------
    # 4) Compute variance (standard deviation) per row.
    # ------------------------------------------------------------------
    diff = a - mean[:, None]                     # (BLOCK_SIZE, FEATURE_SIZE)
    var = tl.sqrt(tl.sum(diff * diff, axis=1) / FEATURE_SIZE)

    # ------------------------------------------------------------------
    # 5) Normalisation: (A - mean) * gamma / (var + 1e‑5) + beta
    # ------------------------------------------------------------------
    # Re‑compute (A - mean) because the CUDA code does it again after variance.
    diff = a - mean[:, None]

    # Load gamma and beta (shape [FEATURE_SIZE])
    gamma_vec = tl.load(gamma + col_offset)
    beta_vec  = tl.load(beta  + col_offset)

    # Apply scale (gamma)
    diff = diff * gamma_vec[None, :]                     # broadcast gamma

    # Normalise by variance + epsilon
    diff = diff / (var[:, None] + 1e-5)                  # broadcast variance

    # Add bias (beta)
    out = diff + beta_vec[None, :]

    # ------------------------------------------------------------------
    # 6) Write the result back to B.
    # ------------------------------------------------------------------
    tl.store(B + offsets, out, mask=valid[:, None])


def triton_kernel(
    A: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    B: torch.Tensor,
    batch_size: int,
    seq_length: int,
    d_model: int,
):
    """
    Entry‑point that mirrors the original ``cuda_kernel`` signature.
    ``d_model`` is ignored because the kernel assumes a fixed feature size of 128.
    """
    # ------------------------------------------------------------------
    # Input validation (mirrors typical CUDA expectations)
    # ------------------------------------------------------------------
    assert A.is_cuda and gamma.is_cuda and beta.is_cuda and B.is_cuda, \
        "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and gamma.dtype == torch.float32 \
           and beta.dtype == torch.float32 and B.dtype == torch.float32, \
        "All tensors must be float32"

    # Ensure contiguous memory layout (required for pointer arithmetic)
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()
    if not beta.is_contiguous():
        beta = beta.contiguous()

    total_len = batch_size * seq_length
    block_size = 4                                 # matches CUDA blockDim.x
    num_blocks = (total_len + block_size - 1) // block_size

    # ------------------------------------------------------------------
    # Launch the Triton kernel.
    # ------------------------------------------------------------------
    _triton_kernel_impl[(num_blocks,)](
        A,
        gamma,
        beta,
        B,
        total_len,
        BLOCK_SIZE=block_size,
        FEATURE_SIZE=128,
    )
    # Synchronise to make the kernel completion observable in tests
    torch.cuda.synchronize()