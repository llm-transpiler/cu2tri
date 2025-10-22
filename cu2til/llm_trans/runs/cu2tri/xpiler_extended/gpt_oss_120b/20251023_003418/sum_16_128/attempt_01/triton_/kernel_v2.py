import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr,
    output_ptr,
    rows,
    inner,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < inner

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    r = 0
    while r < rows:
        ptr = input_ptr + r * inner + offsets
        val = tl.load(ptr, mask=mask, other=0.0)
        acc += val        r += 1

    tl.store(output_ptr + offsets, acc, mask=mask)

def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    inner: int,
):
    """
    Triton wrapper matching the original CUDA kernel signature.
    Performs: output[idx] = sum_{r=0}^{rows-1} input[r * inner + idx]
    """
    if not (input.is_cuda and output.is_cuda):
        raise RuntimeError("input and output must be CUDA tensors")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("Only float32 tensors are supported")
    if not (input.is_contiguous() and output.is_contiguous()):
        raise RuntimeError("Tensors must be contiguous")

    BLOCK_SIZE = 256
    grid_x = (inner + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_x,)

    _triton_kernel_impl[grid](
        input,
        output,
        rows,
        inner,
        BLOCK_SIZE=BLOCK_SIZE,
        # Optional performance tuning:
        # num_warps=4,
        # num_stages=2,
    )