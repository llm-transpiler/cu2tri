import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: row‑wise L2‑normalization (equivalent to the CUDA kernel)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr, B_ptr,               # input and output pointers
    size_1, size_2,             # matrix dimensions (rows, cols)
    eps,                        # epsilon for numerical stability
    NUM_ITER: tl.constexpr       # number of column‑wise tiles (compile‑time)
):
    pid = tl.program_id(0)                 # row index
    row_mask = pid < size_1                 # guard against out‑of‑bounds rows

    # Offsets for the columns processed by this program (128‑wide)
    col_offsets = tl.arange(0, 128)         # [0, 1, ..., 127]

    # ------------------------------------------------------------------
    # First pass: compute sum of squares of the row
    # ------------------------------------------------------------------
    sum_val = tl.zeros([], dtype=tl.float32)   # scalar accumulator
    for i in range(NUM_ITER):
        cur_col = i * 128 + col_offsets
        mask = (cur_col < size_2) & row_mask
        a = tl.load(A_ptr + pid * size_2 + cur_col,
                    mask=mask, other=0.0)      # masked load with zero padding
        sum_val += tl.sum(a * a)                # reduction across the tile

    # ------------------------------------------------------------------
    # Compute scaling factor: 1 / sqrt(mean + eps)
    # ------------------------------------------------------------------
    mean = sum_val / size_2
    scale = 1.0 / tl.sqrt(mean + eps)

    # ------------------------------------------------------------------
    # Second pass: write normalized values to B
    # ------------------------------------------------------------------
    for i in range(NUM_ITER):
        cur_col = i * 128 + col_offsets
        mask = (cur_col < size_2) & row_mask
        a = tl.load(A_ptr + pid * size_2 + cur_col,
                    mask=mask, other=0.0)
        b = a * scale
        tl.store(B_ptr + pid * size_2 + cur_col, b, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, size_1: int, size_2: int):
    """
    Triton implementation of the CUDA kernel that normalizes each row of A
    and writes the result into B. Signature matches the original CUDA kernel.
    """
    # ------------------------------------------------------------------
    # Input validation (mirrors expectations of the original kernel)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda, "A and B must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 supported"
    assert A.shape == (size_1, size_2) and B.shape == (size_1, size_2), \
        "Tensor shapes must match the provided dimensions"
    assert A.is_contiguous() and B.is_contiguous(), "Tensors must be contiguous"

    # ------------------------------------------------------------------
    # Kernel launch configuration
    # ------------------------------------------------------------------
    BLOCK_SIZE = 128
    num_iter = (size_2 + BLOCK_SIZE - 1) // BLOCK_SIZE   # compile‑time tile count
    grid = (size_1,)                                      # one program per row

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