import torch
import triton
import triton.language as tl

# Triton kernel that computes the average across `rows` for each element in the `inner` dimension.
# The kernel is named `_triton_kernel_impl` as required.
@triton.jit
def _triton_kernel_impl(
    input_ptr,          # *float32, input tensor (flattened rows*inner)
    output_ptr,         # *float32, output tensor (inner)
    rows,               # int32, number of rows to reduce over
    inner,              # int32, size of inner dimension
    BLOCK_SIZE: tl.constexpr  # compile‑time block size (threads per program)
):
    # Program ID identifies which block of the inner dimension we are processing.
    pid = tl.program_id(0)
    # Offsets for the threads in this block.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask out threads that are out of bounds (inner may not be a multiple of BLOCK_SIZE).
    mask = offsets < inner

    # Initialize accumulator.
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over rows, accumulating the sum for each inner index.
    r = 0
    while r < rows:
        # Load a vector of values from the current row.
        # The address is: base + r * inner + offsets
        val = tl.load(input_ptr + r * inner + offsets, mask=mask, other=0.0)
        acc += val
        r += 1

    # Compute the average.
    avg = acc / rows

    # Write the result back to the output tensor.
    tl.store(output_ptr + offsets, avg, mask=mask)


def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    inner: int
) -> None:
    """
    Wrapper that launches the Triton kernel `_triton_kernel_impl`.

    Parameters
    ----------
    input : torch.Tensor
        Float32 tensor of shape (rows, inner) stored in row‑major order.
    output : torch.Tensor
        Float32 tensor of shape (inner,) that will receive the averages.
    rows : int
        Number of rows to reduce over.
    inner : int
        Length of the inner dimension (number of elements per row).
    """
    # Basic sanity checks.
    if not input.is_cuda or not output.is_cuda:
        raise RuntimeError("Both input and output tensors must reside on the CUDA device.")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("Tensors must be of type torch.float32.")
    if input.shape != (rows, inner):
        raise RuntimeError(f"Input shape {input.shape} does not match (rows={rows}, inner={inner}).")
    if output.shape != (inner,):
        raise RuntimeError(f"Output shape {output.shape} does not match (inner={inner}).")
    if not input.is_contiguous():
        input = input.contiguous()
    if not output.is_contiguous():
        output = output.contiguous()

    # Triton launch configuration.
    BLOCK_SIZE = 256  # matches the original CUDA block size.
    grid = ((inner + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the kernel.
    _triton_kernel_impl[grid](
        input,
        output,
        rows,
        inner,
        BLOCK_SIZE=BLOCK_SIZE
    )
    # Optional: synchronize for debugging (remove in production for better performance).
    # torch.cuda.synchronize()