import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: row-wise L2 normalization (identical to the CUDA kernel)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr, B_ptr,               # pointers to input and output matrices
    size_1, size_2,             # matrix dimensions (rows, cols)
    eps,                        # epsilon for numerical stability
    NUM_ITER: tl.constexpr      # number of column‑wise iterations (compile‑time)
):
    # ------------------------------------------------------------------
    # Each program (kernel instance) processes one matrix row.
    # ------------------------------------------------------------------
    pid = tl.program_id(0)                 # row index
    row_mask = pid < size_1                 # guard against out‑of‑bounds rows

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    BLOCK_N = 128                           # threads per program (vector width)
    col_offsets = tl.arange(0, BLOCK_N)     # [0, 1, ..., BLOCK_N-1]

    # ------------------------------------------------------------------
    # First pass: compute sum of squares of the row
    # ------------------------------------------------------------------
    sum_val = tl.zeros([], dtype=tl.float32)   # scalar accumulator
    for i in range(NUM_ITER):
        offset = i * BLOCK_N
        cur_col = offset + col_offsets
        mask = (cur_col < size_2) & row_mask   # valid elements in this tile
        a = tl.load(A_ptr + pid * size_2 + cur_col,
                    mask=mask, other=0.0)      # load with zero padding for masked lanes
        sum_val += tl.sum(a * a)                # reduction across the BLOCK_N lanes

    # ------------------------------------------------------------------
    # Compute scaling factor: 1 / sqrt(mean + eps)
    # ------------------------------------------------------------------
    mean = sum_val / size_2
    scale = 1.0 / tl.sqrt(mean + eps)

    # ------------------------------------------------------------------
    # Second pass: write normalized values to B
    # ------------------------------------------------------------------
    for i in range(NUM_ITER):
        offset = i * BLOCK_N
        cur_col = offset + col_offsets
        mask = (cur_col < size_2) & row_mask
        a = tl.load(A_ptr + pid * size_2 + cur_col,
                    mask=mask, other=0.0)
        b = a * scale
        tl.store(B_ptr + pid * size_2 + cur_col, b, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper that mimics the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, size_1: int, size_2: int):
    """
    Triton implementation of the CUDA kernel that normalizes each row of A
    and writes the result into B.

    Parameters
    ----------
    A : torch.Tensor
        Input matrix of shape (size_1, size_2), dtype torch.float32, CUDA.
    B : torch.Tensor
        Output matrix of the same shape and dtype as A.
    size_1 : int
        Number of rows.
    size_2 : int
        Number of columns.
    """
    # ------------------------------------------------------------------
    # Basic sanity checks (mirrors expectations of the original kernel)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda, "A and B must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 supported"
    assert A.shape == (size_1, size_2) and B.shape == (size_1, size_2), \
        "Tensor shapes must match the provided dimensions"
    assert A.is_contiguous() and B.is_contiguous(), "Tensors must be contiguous"

    # ------------------------------------------------------------------
    # Kernel launch configuration
    # ------------------------------------------------------------------
    BLOCK_N = 128                                 # threads per program (must match kernel)
    num_iter = (size_2 + BLOCK_N - 1) // BLOCK_N   # compile‑time iteration count
    grid = (size_1,)                               # one program per row

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A,                     # input pointer
        B,                     # output pointer
        size_1,
        size_2,
        1e-5,                  # eps
        NUM_ITER=num_iter,     # compile‑time constant
        num_warps=4,           # 128 threads = 4 warps
    )