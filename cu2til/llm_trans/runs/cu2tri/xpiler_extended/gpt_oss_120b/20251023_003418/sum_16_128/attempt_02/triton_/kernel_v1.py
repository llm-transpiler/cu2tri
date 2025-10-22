# Triton implementation of the reduction kernel
import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    input_ptr,          # const float* __restrict__ input
    output_ptr,         # float* __restrict__ output
    rows,               # int rows
    inner,              # int inner
    BLOCK_SIZE: tl.constexpr,  # compile‑time block size
):
    """
    For each index `idx` in [0, inner):
        output[idx] = sum_{r=0}^{rows-1} input[r * inner + idx]
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices
    mask = offsets < inner                      # guard against OOB

    # accumulator for the reduction
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # loop over rows (runtime bound)
    r = 0
    while r < rows:
        # Load a strided element from each row
        # address = input_ptr + r * inner + offsets
        acc += tl.load(input_ptr + r * inner + offsets,
                       mask=mask,
                       other=0.0)
        r += 1

    # Write the result back
    tl.store(output_ptr + offsets, acc, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function – entry point with the same signature as the CUDA code
# ----------------------------------------------------------------------
def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    inner: int,
) -> None:
    """
    Launches the Triton reduction kernel.

    Parameters
    ----------
    input : torch.Tensor
        2‑D tensor of shape (rows, inner) with dtype torch.float32 on CUDA.
    output : torch.Tensor
        1‑D tensor of shape (inner,) with dtype torch.float32 on CUDA.
    rows : int
        Number of rows in `input`.
    inner : int
        Length of the reduction dimension (the second dimension of `input`).
    """
    # Basic sanity checks
    if not (input.is_cuda and output.is_cuda):
        raise RuntimeError("Both input and output tensors must reside on the CUDA device.")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("Only torch.float32 tensors are supported.")
    if input.dim() != 2 or output.dim() != 1:
        raise RuntimeError("Expected input shape (rows, inner) and output shape (inner,).")
    if input.shape[0] != rows or input.shape[1] != inner:
        raise ValueError("Tensor shapes do not match the provided `rows` and `inner` arguments.")
    if output.shape[0] != inner:
        raise ValueError("Output tensor length must equal `inner`.")

    # Choose a block size that matches the CUDA implementation (256 threads)
    BLOCK_SIZE = 256

    # Compute grid size: one program per BLOCK_SIZE elements of `inner`
    grid = lambda meta: ((inner + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    # Launch the kernel
    _triton_kernel_impl[grid](
        input,
        output,
        rows,
        inner,
        BLOCK_SIZE=BLOCK_SIZE,
    )