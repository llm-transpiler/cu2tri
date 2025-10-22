import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    A_ptr, B_ptr,
    size_1, size_2,
    eps,
    BLOCK_SIZE: tl.constexpr,
    N: tl.constexpr,
):
    """
    Triton kernel that normalizes each row of A and writes the result to B.
    Each program (row) processes its entire row in blocks of `BLOCK_SIZE`.
    """
    pid = tl.program_id(0)  # row index
    if pid >= size_1:
        return

    # Pointers to the beginning of the current row
    row_offset = pid * size_2
    A_row_ptr = A_ptr + row_offset
    B_row_ptr = B_ptr + row_offset

    # -------------------------------------------------
    # Compute sum of squares for the row (mean of squares)
    # -------------------------------------------------
    sum_sq = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        col = tl.arange(0, BLOCK_SIZE) + offset          # [BLOCK_SIZE]
        mask = col < size_2                               # mask out-of-range
        a = tl.load(A_row_ptr + col, mask=mask, other=0.0)  # load block
        sum_sq = sum_sq + tl.sum(a * a)                    # accumulate

    # -------------------------------------------------
    # Compute scaling factor: 1 / sqrt(mean + eps)
    # -------------------------------------------------
    mean = sum_sq / size_2
    scale = 1.0 / tl.sqrt(mean + eps)

    # -------------------------------------------------
    # Write normalized values to B
    # -------------------------------------------------
    for offset in range(0, N, BLOCK_SIZE):
        col = tl.arange(0, BLOCK_SIZE) + offset
        mask = col < size_2
        a = tl.load(A_row_ptr + col, mask=mask, other=0.0)
        b = a * scale
        tl.store(B_row_ptr + col, b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, size_1: int, size_2: int):
    """
    Wrapper that launches the Triton kernel with the same semantics as the original CUDA kernel.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size_1, size_2), dtype torch.float32, on CUDA.
    B : torch.Tensor
        Output tensor of the same shape and dtype as A.
    size_1 : int
        Number of rows.
    size_2 : int
        Number of columns.
    """
    # ------------------------------------------------------------------
    # Basic validation (mirrors the expectations of the original CUDA code)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda, "A and B must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 tensors are supported"
    assert A.shape == (size_1, size_2), f"A shape mismatch: expected ({size_1}, {size_2}), got {A.shape}"
    assert B.shape == (size_1, size_2), f"B shape mismatch: expected ({size_1}, {size_2}), got {B.shape}"

    # Ensure contiguous memory layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()

    BLOCK_SIZE = 256               # threads per program (must be a multiple of 32)
    grid = (size_1,)               # one program per row

    _triton_kernel_impl[grid](
        A,
        B,
        size_1,
        size_2,
        1e-5,                      # eps
        BLOCK_SIZE=BLOCK_SIZE,
        N=size_2,                  # compile‑time constant for loop bounds
        num_warps=BLOCK_SIZE // 32,
    )