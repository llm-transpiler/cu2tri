import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr,   # *float32
    output_ptr,  # *float32
    total,       # i32
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.math.sin(x)
    tl.store(output_ptr + offsets, y, mask=mask)

def triton_kernel(input: torch.Tensor, output: torch.Tensor, total: int):
    """
    Triton implementation of the sin kernel.
    Mirrors the signature of the original CUDA kernel.
    """
    if not (input.is_cuda and output.is_cuda):
        raise ValueError("input and output must be CUDA tensors")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise ValueError("Only float32 tensors are supported")
    if not (input.is_contiguous() and output.is_contiguous()):
        raise ValueError("Tensors must be contiguous")
    if total > input.numel() or total > output.numel():
        raise ValueError("total exceeds tensor size")
    BLOCK_SIZE = 256
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    _triton_kernel_impl[grid](
        input,
        output,
        total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.cuda.synchronize()