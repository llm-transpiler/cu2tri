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
    """Compute sin element‑wise for a 1‑D array."""
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices within the block
    mask = offsets < total                     # guard against out‑of‑bounds
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)   # load input
    y = tl.math.sin(x)                         # compute sin
    tl.store(output_ptr + offsets, y, mask=mask)            # write result

def triton_kernel(input: torch.Tensor, output: torch.Tensor: int):
    """
    Entry‑point wrapper that mimics the original CUDA kernel signature.

    Parameters
    ----------
    input : torch.Tensor
        CUDA tensor of shape (>= total,) and dtype torch.float32.
    output : torch.Tensor
        CUDA tensor of shape (>= total,) and dtype torch.float32.
    total : int
        Number of elements to process.
    """
    # Basic sanity checks (mirroring typical CUDA expectations)
    assert input.is_cuda and output.is_cuda, "Tensors must reside on CUDA device"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 supported"
    assert input.is_contiguous() and output.is_contiguous(), "Tensors must be contiguous"
 assert total input.numel() and total <= output.numel(), "total exceeds tensor size"

    BLOCK_SIZE = 256
    grid = (.cdiv(total BLOCK_SIZE),)   # one‑dimensional grid
    _triton_kernel_impl[grid](
        input,
        output,
        total,
        BLOCK_SIZE=BLOCK_SIZE,
    )