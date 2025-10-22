import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Constants (match the CUDA implementation)
# ----------------------------------------------------------------------
FEATURE_SIZE = 128          # d_model is hard‑coded to 128 in the CUDA kernel
EPS = 1e-5                  # epsilon used in the denominator

@triton.jit
def _triton_kernel_impl(
    A,          # *float32, shape [batch_seq_len * FEATURE_SIZE]
    gamma,      # *float32, shape [FEATURE_SIZE]
    beta,       # *float32, shape [FEATURE_SIZE]
    B,          # *float32, shape [batch_seq_len * FEATURE_SIZE]
    total_len,  # int32, total number of rows = batch_size * seq_length
    BLOCK_SIZE: tl.constexpr,   # threads per program (matches CUDA blockDim.x)
):
    """
    Triton implementation of the original CUDA kernel.
    Each Triton program processes BLOCK_SIZE rows (identical to a CUDA block).
    The original CUDA kernel also restricts processing to idx < 4, which we
    preserve via an additional mask.
    """
    # ------------------------------------------------------------------
    # 1) Compute logical row indices for this program.
    # ------------------------------------------------------------------
    row_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask for rows that belong to the workload.
    # Original CUDA kernel also checks idx < 4.
    valid = (row_idx < total_len) & (row_idx < 4)

    # ------------------------------------------------------------------
    # 2) Load the 128‑element vectors for the selected rows.
    # ------------------------------------------------------------------
    col_offset = tl.arange(0, FEATURE_SIZE)                     # [0, 1, ..., 127]
    offsets = row_idx[:, None] * FEATURE_SIZE + col_offset[None, :]  # (BLOCK_SIZE, FEATURE_SIZE)

    # Load A; out‑of‑range elements are filled with 0.0 (masked later)
    a = tl.load(A + offsets, mask=valid[:, None], other=0.0)

    # ------------------------------------------------------------------
    # 3) Compute mean per row.
    # ------------------------------------------------------------------
    mean = tl.sum(a, axis=1) / FEATURE_SIZE

    # ------------------------------------------------------------------
    # 4) Compute variance per row.
    # ------------------------------------------------------------------
    diff = a - mean[:, None]                     # (BLOCK_SIZE, FEATURE_SIZE)
    var = tl.sqrt(tl.sum(diff * diff, axis=1) / FEATURE_SIZE)

    # ------------------------------------------------------------------
    # 5) Normalisation: (A - mean) * gamma / (var + EPS) + beta
    # ------------------------------------------------------------------
    # Re‑compute (A - mean) because the CUDA code does it again after variance.
    diff = a - mean[:, None]

    # Load gamma and beta (shape [FEATURE_SIZE]) once per program.
    gamma_vec = tl.load(gamma + col_offset)
    beta_vec  = tl.load(beta  + col_offset)

    # Apply scale (gamma) and normalisation.
    diff = diff * gamma_vec[None, :]                     # broadcast gamma
    diff = diff / (var[:, None] + EPS)                  # broadcast variance

    # Add bias (beta).
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
    ``d_model`` is ignored because the kernel hard‑codes 128 features.
    """
    # ------------------------------------------------------------------
    # Input validation (mirrors typical CUDA expectations)
    # ------------------------------------------------------------------
    assert A.is_cuda and gamma.is_cuda and beta.is_cuda and B.is_cuda, \
        "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and gamma.dtype == torch.float32 \
           and beta.dtype == torch.float32 and B.dtype == torch.float32, \
        "All tensors must be float32"

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
    )