import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr,          # *float32
    output_ptr,         # *float32
    rows: tl.int32,
    inner: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < inner

    # Initialize max values to -inf
    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)

    i = 0
    while i < rows:
        # Linear index of element (i, offsets) in the flattened input
        idx = i * inner + offsets
        cur = tl.load(input_ptr + idx, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, cur)
        i += 1

    tl.store(output_ptr + offsets, max_val, mask=mask)

def triton_kernel(input: torch.Tensor, output: torch.Tensor, rows: int, inner: int):
    """
    Triton implementation of the reduction kernel.
    Computes, for each column index `idx` in [0, inner), the maximum value across `rows` rows.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor containing `rows * inner` elements. Can be 1‑D (flattened) or 2‑D
        with shape `(rows, inner)`. Must be float32 and reside on CUDA.
    output : torch.Tensor
        Output tensor of length `inner`. Must be float32 and reside on CUDA.
    rows : int
        Number of logical rows in the input.
    inner : int
        Number of columns (inner dimension).
    """
    # Type and device checks
    if not isinstance(input, torch.Tensor) or not isinstance(output, torch.Tensor):
        raise TypeError("input and output must be torch.Tensor")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise TypeError("input and output must be float32 tensors")
    if not input.is_cuda or not output.is_cuda:
        raise RuntimeError("input and output must be CUDA tensors")

    # Validate element counts
    expected_numel = rows * inner
    if input.numel() != expected_numel:
        raise ValueError(f"input numel {input.numel()} does not match rows*inner ({expected_numel})")
    if output.numel() != inner:
        raise ValueError(f"output numel {output.numel()} does not match inner ({inner})")

    # Ensure contiguous memory layout
    input = input.contiguous()
    output = output.contiguous()

    BLOCK_SIZE = 256
    grid = ((inner + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    _triton_kernel_impl[grid](
        input,
        output,
        rows,
        inner,
        BLOCK_SIZE=BLOCK_SIZE
    )