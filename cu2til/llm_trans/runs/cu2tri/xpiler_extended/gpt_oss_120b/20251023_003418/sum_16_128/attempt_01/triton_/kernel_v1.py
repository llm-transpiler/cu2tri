import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr,
    output_ptr,
    rows,
    inner,
    BLOCK_SIZE: tl.constexpr
):
    # Program (block) ID
    pid = tl.program_id(0)
    # Offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to avoid OOB accesses
    mask = offsets < inner

    # Accumulator for the reduction
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over rows and accumulate
    r = 0
    while r < rows:
        # Pointer to the element at (r,)
        ptr = input_ptr + r * inner + offsets
        # Load with mask (zero for OOB)
        val = tl.load(ptr, mask=mask, other=0.0)
        acc += val
        r += 1

    # Write the result back
    tl.store(output_ptr + offsets, acc, mask=mask)

def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    rows: int,
   : int
):
    """
    Triton wrapper matching the original CUDA kernel signature.
    Performs: output[idx] = sum_{r=0}^{rows-1} input[r * inner + idx]
    """
    # Basic sanity checks
    assert input.is_cuda and output.is_cuda, "Tensors must be CUDA tensors"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 supported"
    assert input.is_contiguous() and output.is_contiguous(), "Tensors must be contiguous"

    BLOCK_SIZE = 256
    grid_x = (inner + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_x,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        output,
        rows,
        inner,
        BLOCK_SIZE=BLOCK_SIZE
    )