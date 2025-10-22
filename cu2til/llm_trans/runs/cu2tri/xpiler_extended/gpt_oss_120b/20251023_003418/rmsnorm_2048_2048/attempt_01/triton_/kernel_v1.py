import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, size_1, size_2, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # Row index
    if pid >= size_1:
        return

    # Base pointers for the current row
    row_offset = pid * size_2
    A_row_ptr = A_ptr + row_offset
    B_row_ptr = B_ptr + row_offset

    # Compute sum of squares for the row
    sum_sq = tl.float32(0.0)
    offset = 0
    while offset < size_2:
        col = tl.arange(0, BLOCK_SIZE) + offset
        mask = col < size_2
        a = tl.load(A_row_ptr + col, mask=mask, other=0.0)
        # Reduce across the block (threads) to get sum of squares for this chunk
        sum_sq += tl.sum(a * a)
        offset += BLOCK_SIZE

    # Compute scaling factor
    mean = sum_sq / tl.float32(size_2)
    scale = 1.0 / tl.sqrt(mean + eps)

    # Write normalized values to B
    offset = 0
    while offset < size_2:
        col = tl.arange(0, BLOCK_SIZE) + offset
        mask = col < size_2
        a = tl.load(A_row_ptr + col, mask=mask, other=0.0)
        b = a * scale
        tl.store(B_row_ptr + col, b, mask=mask)
        offset += BLOCK_SIZE

def triton_kernel(A: torch.Tensor, B: torch.Tensor, size_1: int, size_2: int):
    """
    Triton implementation of the CUDA kernel that normalizes each row of A
    and writes the result to B.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size_1, size_2), dtype torch.float32, CUDA device.
    B : torch.Tensor
        Output tensor of the same shape and dtype as A.
    size_1 : int
        Number of rows.
    size_2 : int
        Number of columns.
    """
    assert A.is_cuda and B.is_cuda, "A and B must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 tensors are supported"
    assert A.shape == (size_1, size_2), f"A shape mismatch: expected ({size_1}, {size_2}), got {A.shape}"
    assert B.shape == (size_1, size_2), f"B shape mismatch: expected ({size_1}, {size_2}), got {B.shape}"
    # Ensure contiguous layout
    A = A.contiguous()
    B = B.contiguous()

    BLOCK_SIZE = 256  # threads per program (must be multiple of 32)
    grid = (size_1,)
    _triton_kernel_impl[grid](
        A,
        B,
        size_1,
        size_2,
        1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32,
    )