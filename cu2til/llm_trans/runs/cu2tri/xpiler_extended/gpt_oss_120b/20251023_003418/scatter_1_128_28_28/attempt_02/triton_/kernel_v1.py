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
    Scatter kernel: for each (row, w) pair, write input[row, w] to
    output[row, indices[row, w]].
    """
    total = rows * W
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    # Row and column within the 2‑D view
    row = offsets // W
    col = offsets % W

    # Linear offset for the input and indices tensors
    idx = row * W + col

    # Load the source value and its target column index
    input_val = tl.load(input_ptr + idx, mask=mask, other=0.0)
    index_val = tl.load(indices_ptr + idx, mask=mask, other=0)

    # Destination offset in the output tensor
    out_idx = row * W + index_val

    # Store the scattered value
    tl.store(output_ptr + out_idx, input_val, mask=mask)

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
    Entry‑point that mimics the original CUDA kernel signature.
    Performs a device‑to‑device copy followed by the scatter operation.
    """
    # Basic sanity checks
    if not (input.is_cuda and indices.is_cuda and output.is_cuda):
        raise RuntimeError("All tensors must reside on the CUDA device")
    if input.dtype != torch.float32:
        raise TypeError("`input` must be a torch.float32 tensor")
    if indices.dtype != torch.int32:
        raise TypeError("`indices` must be a torch.int32 tensor")
    if output.dtype != torch.float32:
        raise TypeError("`output` must be a torch.float32 tensor")

    # Equivalent of cudaMemcpyDeviceToDevice
    output.copy_(input)

    rows = N * C * H
    total = rows * W
    BLOCK_SIZE = 256  # Tune for the target GPU if needed

    # Compute grid dimensions
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        indices,
        output,
        rows,
        W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Optional synchronization (usually not required)
    # torch.cuda.synchronize()