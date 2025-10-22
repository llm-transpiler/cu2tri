import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr,
    indices_ptr,
    output_ptr,
    rows,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scatter kernel: output[row, indices[row, w]] = input[row, w]
    """
    pid_row = tl.program_id(0)          # row index
    pid_col = tl.program_id(1)          # column block index

    # Guard against out‑of‑range rows (should never trigger)
    row_mask = pid_row < rows

    # Offsets of the columns handled by this program instance
    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < W

    # Combined mask for valid loads/stores
    mask = row_mask & col_mask

    # Linear offsets for the input and index tensors
    input_offset = pid_row * W + col_offsets
    indices_offset = pid_row * W + col_offsets

    # Load the source values and their destination indices
    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    idx_val = tl.load(indices_ptr + indices_offset, mask=mask, other=0)

    # Compute where each value should be written in the output
    output_offset = pid_row * W + idx_val

    # Ensure we only write to valid output positions
    store_mask = mask & (idx_val < W)

    tl.store(output_ptr + output_offset, input_val, mask=store_mask)


def triton_kernel(
    input: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    H: int,
    W: int,
):
    """
    Triton implementation of the original CUDA scatter kernel.

    Parameters
    ----------
    input : torch.Tensor
        Float tensor on CUDA device, shape (N*C*H, W) or any contiguous view.
    indices : torch.Tensor
        Int tensor of the same shape as ``input``.
    output : torch.Tensor
        Float tensor on CUDA device where the result will be stored.
    N, C, H, W : int
        Original 4‑D dimensions.
    """
    if not (input.is_cuda and indices.is_cuda and output.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    # Ensure contiguous memory layout for pointer arithmetic
    input = input.contiguous()
    indices = indices.contiguous()
    output = output.contiguous()

    # Device‑to‑device copy (mirrors the original cudaMemcpy)
    output.copy_(input, non_blocking=True)

    rows = N * C * H
    BLOCK_SIZE = 128  # Tunable; 128 works well on Hopper GPUs

    grid = (rows, (W + BLOCK_SIZE - 1) // BLOCK_SIZE)

    _triton_kernel_impl[grid](
        input,
        indices,
        output,
        rows,
        W,
        BLOCK_SIZE,
        num_warps=4,   # 4 warps × 32 threads = 128 threads per program
    )
    return output