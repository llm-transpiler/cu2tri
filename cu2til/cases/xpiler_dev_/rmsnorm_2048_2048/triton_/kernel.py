import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel_impl(
    A_ptr,
    B_ptr,
    # The number of columns in the matrix.
    N_COLS: tl.constexpr,
    # The stride to move from one row to the next in the input matrix.
    stride_a_row,
    # The stride to move from one row to the next in the output matrix.
    stride_b_row,
    # The block size for the column dimension.
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton kernel for row-wise RMS normalization.
    Each program instance processes one row of the input matrix A.
    """
    # Get the row index for the current program instance.
    row_idx = tl.program_id(axis=0)

    # --- Step 1: Load a row of data from A ---
    # Create pointers to the start of the current row in A and B.
    row_start_ptr_A = A_ptr + row_idx * stride_a_row
    row_start_ptr_B = B_ptr + row_idx * stride_b_row

    # Create a range of column indices.
    # e.g., [0, 1, 2, ..., BLOCK_SIZE_N-1]
    cols = tl.arange(0, BLOCK_SIZE_N)
    
    # Create a mask to safely handle rows where N_COLS is not a multiple of BLOCK_SIZE_N.
    # For this specific problem, N_COLS=2048 and BLOCK_SIZE_N will be 2048, so the mask is always true.
    mask = cols < N_COLS

    # Load the row from A. Use the mask to prevent out-of-bounds memory access.
    # Padded values are set to 0.0, which won't affect the sum of squares.
    a = tl.load(row_start_ptr_A + cols, mask=mask, other=0.0).to(tl.float32)

    # --- Step 2: Compute the scaling factor ---
    # Constant for numerical stability.
    eps = 1e-5

    # Calculate the sum of squares for the row.
    # The reduction is performed in parallel across the `cols` dimension (axis=0).
    sum_sq = tl.sum(a * a, axis=0)

    # Calculate the mean of the squares.
    mean = sum_sq / N_COLS

    # Calculate the reciprocal of the square root of the mean (rsqrt).
    # tl.math.rsqrt is generally more efficient than 1.0 / tl.sqrt.
    scale = tl.math.rsqrt(mean + eps)

    # --- Step 3: Normalize the row and store it in B ---
    # Apply the scaling factor to the original row data.
    b = a * scale

    # Store the normalized row back to the B matrix.
    tl.store(row_start_ptr_B + cols, b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, size_1: int, size_2: int):
    """
    Wrapper function for the Triton RMS Normalization kernel.

    This function serves as the entry point, handling grid configuration and launching
    the Triton kernel. It has a signature analogous to the original CUDA wrapper.

    Args:
        A (torch.Tensor): Input tensor of shape (size_1, size_2) and dtype float32.
        B (torch.Tensor): Output tensor of shape (size_1, size_2) and dtype float32.
        size_1 (int): The number of rows in the tensors.
        size_2 (int): The number of columns in the tensors.
    """
    # Input validation to ensure correctness.
    assert A.shape == (size_1, size_2), f"Input A shape mismatch: expected ({size_1}, {size_2}), got {A.shape}"
    assert B.shape == (size_1, size_2), f"Output B shape mismatch: expected ({size_1}, {size_2}), got {B.shape}"
    assert A.is_cuda and B.is_cuda, "Tensors must be on a CUDA device"
    assert A.is_contiguous() and B.is_contiguous(), "Tensors must be contiguous"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Tensors must be of type float32"

    # Configure the launch grid. We launch one program instance for each row.
    grid = (size_1,)

    # Determine the block size for the column dimension.
    # Using triton.next_power_of_2 is a good practice for performance.
    # For the given problem size of 2048, this will be 2048.
    BLOCK_SIZE_N = triton.next_power_of_2(size_2)

    # Launch the Triton kernel.
    _triton_kernel_impl[grid](
        A,
        B,
        N_COLS=size_2,
        stride_a_row=A.stride(0),
        stride_b_row=B.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )