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

    # Initialize max values to -inf (acts like -FLT_MAX for the reduction)
    max_val = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)

    # Dynamic loop over rows
    r = tl.zeros([], dtype=tl.int32)
    while r < rows:
        idx = r * inner + offsets
        val = tl.load(input_ptr + idx, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, val)
        r += 1

    tl.store(output_ptr + offsets, max_val, mask=mask)


def triton_kernel(input: torch.Tensor, output: torch.Tensor, rows: int, inner: int):
    """
    Triton implementation of the reduction kernel.
    Parameters
    ----------
    input : torch.Tensor
        2‑D tensor of shape (rows, inner) and dtype torch.float32 on CUDA device.
    output : torch.Tensor
        1‑D tensor of shape (inner,) and dtype torch.float32 on the same CUDA device.
    rows : int
        Number of rows in `input`.
    inner : int
        Length of the inner dimension (number of columns).
    """
    assert input.is_cuda and output.is_cuda, "Tensors must be on CUDA device"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 supported"
    assert input.shape == (rows, inner), f"Expected input shape ({rows}, {inner}), got {input.shape}"
    assert output.shape == (inner,), f"Expected output shape ({inner},), got {output.shape}"

    BLOCK_SIZE = 256
    grid = ((inner + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _triton_kernel_impl[grid](
        input,
        output,
        rows,
        inner,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )