import torch
import triton
import triton.language as tl
import math

@triton.jit
def _triton_kernel_impl(
    input_ptr,
    output_ptr,
    dim0,
    dim1,
    dim2,
    dim3,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # 1‑D linear indices for this program instance
    offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    offsets = tl.cast(offsets, tl.int64)

    mask = offsets < total

    # Load input elements (masked)
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Decompose linear index into 4‑D coordinates
    idx = offsets
    i3 = idx % dim3
    idx = idx // dim3
    i2 = idx % dim2
    idx = idx // dim2
    i1 = idx % dim1
    idx = idx // dim1
    i0 = idx % dim0

    # Permute axes: (0, 2, 3, 1)
    out0 = i0
    out1 = i2
    out2 = i3
    out3 = i1

    # Compute output linear index using strides derived from dimensions
    stride0 = dim2 * dim3 * dim1
    stride1 = dim3 * dim1
    stride2 = dim1
    out_idx = out0 * stride0 + out1 * stride1 + out2 * stride2 + out3

    # Write to output (masked)
    tl.store(output_ptr + out_idx, input_vals, mask=mask)


def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    dim0: int,
    dim1: int,
    dim2: int,
    dim3: int,
):
    """
    Triton implementation of the CUDA transpose kernel.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape (dim0, dim1, dim2, dim3), contiguous, dtype float32.
    output : torch.Tensor
        Output tensor of shape (dim0, dim2, dim3, dim1), contiguous, dtype float32.
    dim0, dim1, dim2, dim3 : int
        Dimensions of the input tensor.
    """
    total = dim0 * dim1 * dim2 * dim3
    BLOCK_SIZE = 256
    grid = (math.ceil(total / BLOCK_SIZE),)

    # Basic sanity checks
    if not (input.is_cuda and output.is_cuda):
        raise RuntimeError("Both input and output must be CUDA tensors")
    if not (input.is_contiguous() and output.is_contiguous()):
        raise RuntimeError("Both input and output must be contiguous")
    if not (input.dtype == torch.float32 and output.dtype == torch.float32):
        raise RuntimeError("Only float32 tensors are supported")

    _triton_kernel_impl[grid](
        input,
        output,
        dim0,
        dim1,
        dim2,
        dim3,
        total,
        BLOCK_SIZE=BLOCK_SIZE,
    )