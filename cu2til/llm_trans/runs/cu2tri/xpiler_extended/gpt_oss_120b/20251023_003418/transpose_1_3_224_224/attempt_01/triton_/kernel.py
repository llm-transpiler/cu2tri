import torch
import triton
import triton.language as tl

# Triton kernel implementation
@triton.jit
def _triton_kernel_impl(
    input_ptr,
    output_ptr,
    dim0,
    dim1,
    dim2,
    dim3,
    total,
    BLOCK_SIZE: tl.constexpr
):
    # Linear thread IDs
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    idx = offsets

    # Decompose linear index into 4‑D coordinates (row‑major order)
    tmp = idx
    i3 = tmp % dim3
    tmp = tmp // dim3
    i2 = tmp % dim2
    tmp = tmp // dim2
    i1 = tmp % dim1
    tmp = tmp // dim1
    i0 = tmp % dim0

    # Axis permutation: (0, 2, 3, 1)
    out0 = i0
    out1 = i2
    out2 = i3
    out3 = i1

    # Hard‑coded strides exactly as in the original CUDA kernel
    out_idx = out0 * 150528 + out1 * 672 + out2 * 3 + out3

    # Load from input and store to output (masked)
    val = tl.load(input_ptr + idx, mask=mask, other=0.0)
    tl.store(output_ptr + out_idx, val, mask=mask)


def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    dim0: int,
    dim1: int,
    dim2: int,
    dim3: int
):
    """
    Triton entry point mirroring the original CUDA `cuda_kernel`.
    Parameters:
        input  – contiguous float32 tensor on the GPU
        output – contiguous float32 tensor on the GPU (pre‑allocated)
        dim0, dim1, dim2, dim3 – dimensions of the input tensor
    """
    assert input.is_cuda and output.is_cuda, "Tensors must reside on the GPU"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 supported"
    input = input.contiguous()
    output = output.contiguous()

    total = dim0 * dim1 * dim2 * dim3
    BLOCK_SIZE = 256
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _triton_kernel_impl[grid](
        input,
        output,
        dim0,
        dim1,
        dim2,
        dim3,
        total,
        BLOCK_SIZE=BLOCK_SIZE
    )