import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr,          # *float32
    output_ptr,         # *float32
    rows: tl.int32,
    inner: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < inner

    # Initialize max values to -inf (equivalent to -FLT_MAX for reduction)
    max_val = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)

    # Loop over rows
    r = tl.zeros([], dtype=tl.int32)          # scalar row index
    while r < rows:
        # Compute linear index for the current row and each offset
        idx = r * inner + offsets
        # Load values with mask; out-of-range lanes get -inf
        val = tl.load(input_ptr + idx, mask=mask, other=-float('inf'))
        # Reduce max
        max_val = tl.maximum(max_val, val)
        r = r + 1

    # Write result
    tl.store(output_ptr + offsets, max_val, mask=mask)


def triton_kernel(input: torch.Tensor, output: torch.Tensor, rows: int, inner: int):
    """
    Triton implementation of the reduction kernel.
    Parameters
    ----------
    input : torch.Tensor
        Tensor containing `rows * inner` elements of type float32 on a CUDA device.
        Can be either shape (rows, inner) or a flat 1‑D tensor of length rows*inner.
    output : torch.Tensor
        Tensor of shape (inner,) and dtype float32 on the same CUDA device.
    rows : int
        Number of rows in the logical 2‑D view of `input`.
    inner : int
        Length of the inner dimension (number of columns).
    """
    # Basic validation
    assert input.is_cuda and output.is_cuda, "Tensors must be on CUDA device"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 supported"

    # Accept both 2‑D and flat 1‑D input layouts
    if input.dim() == 2:
        assert input.shape == (rows, inner), f"Expected input shape ({rows}, {inner}), got {input.shape}"
    elif input.dim() == 1:
        assert input.numel() == rows * inner, (
            f"Expected flat input of size {rows * inner}, got {input.numel()}"
        )
    else:
        raise AssertionError(f"Input must be 1‑D or 2‑D, got {input.dim()}‑D")

    # Output must be 1‑D of length `inner`
    assert output.dim() == 1 and output.shape[0] == inner, (
        f"Expected output shape ({inner},), got {output.shape}"
    )

    BLOCK_SIZE = 256
    grid = ((inner + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _triton_kernel_impl[grid](
        input,
        output,
        rows,
        inner,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )