import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel_impl(
    A_ptr,
    x_ptr,
    y_ptr,
    stride_a_row,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton kernel to compute y = A @ x for the first 3 rows of A.
    This kernel is a direct translation of the provided CUDA code's logic.
    """
    # Each program instance computes one row of the output vector y.
    # This corresponds to the CUDA logic: int row = blockIdx.x * blockDim.x + threadIdx.x;
    row = tl.program_id(axis=0)

    # The original CUDA kernel has a specific guard condition: if (row < 3).
    # We replicate this exact behavior. It ignores the input 'm' size from the
    # wrapper and only computes the first 3 rows of the output.
    if row < 3:
        # The inner loop runs 16 times in CUDA: for (int i = 0; i < 16; i++)
        # We model this using tl.arange and a constexpr block size.
        col_offsets = tl.arange(0, BLOCK_SIZE_N)

        # Pointers to the current row of A and the vector x.
        # A_ptr + row * stride_a_row gives the start of the row.
        # col_offsets accesses each element in that row.
        a_ptrs = A_ptr + row * stride_a_row + col_offsets
        x_ptrs = x_ptr + col_offsets

        # Load a full row of A and the vector x.
        # The original CUDA code assumes n=16 and does not use a boundary check/mask.
        a_vals = tl.load(a_ptrs)
        x_vals = tl.load(x_ptrs)

        # Compute the dot product: sum += A[row * 16 + i] * x[i];
        # Triton performs this in a vectorized manner.
        sum_val = tl.sum(a_vals * x_vals, axis=0)

        # Write the result to y: y[row] = sum;
        y_ptr_row = y_ptr + row
        tl.store(y_ptr_row, sum_val)


def triton_kernel(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, m: int, n: int):
    """
    Triton wrapper for a matrix-vector multiplication kernel.

    This function provides an entry point with functionality identical to the
    original CUDA code's `cuda_kernel` function. It sets up the grid and launches
    the `_triton_kernel_impl` kernel.

    Args:
        A (torch.Tensor): The input matrix of shape (m, n), must be float32.
        x (torch.Tensor): The input vector of shape (n,), must be float32.
        y (torch.Tensor): The output vector of shape (m,), must be float32.
        m (int): The number of rows in matrix A.
        n (int): The number of columns in matrix A and the size of vector x.
    """
    # --- Argument validation ---
    assert A.is_cuda and x.is_cuda and y.is_cuda, "All tensors must be on a CUDA device."
    assert A.dtype == torch.float32 and x.dtype == torch.float32 and y.dtype == torch.float32, "All tensors must be of type float32."
    assert A.shape == (m, n), f"A must have shape ({m}, {n}), but got {A.shape}"
    assert x.shape == (n,), f"x must have shape ({n},), but got {x.shape}"
    assert y.shape == (m,), f"y must have shape ({m},), but got {y.shape}"
    # The original CUDA kernel hardcodes n=16 in its loop.
    assert n == 16, "The kernel is hardcoded for n=16, matching the CUDA implementation."
    # For simplicity and to match common use cases, assume contiguous tensors.
    # The kernel currently relies on this for pointer arithmetic.
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous(), "All tensors must be contiguous."

    # --- Grid configuration ---
    # The CUDA code launches threads to cover 'm' rows:
    # numBlocks((m + 3 - 1) / 3), blockSize(3)
    # This is conceptually equivalent to launching one "work item" per row up to 'm'.
    # In Triton, we define a 1D grid where each program instance handles one row.
    # We launch 'm' programs, and the kernel uses tl.program_id(0) as the row index.
    # The kernel's `if row < 3` condition ensures only the first 3 do work,
    # mimicking the original CUDA code's behavior.
    grid = (m,)

    # --- Kernel launch ---
    # The inner dimension is hardcoded to 16 in the CUDA kernel's loop.
    # We pass this as a compile-time constant (constexpr).
    BLOCK_SIZE_N = 16

    _triton_kernel_impl[grid](
        A,
        x,
        y,
        A.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )