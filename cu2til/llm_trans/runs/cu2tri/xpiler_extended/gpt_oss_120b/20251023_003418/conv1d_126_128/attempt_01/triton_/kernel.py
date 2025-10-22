import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr,   # *float
    kernel_ptr,  # *float
    output_ptr,  # *float
    output_size: tl.int32,   # runtime bound to avoid OOB writes
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Original CUDA kernel processes idx < 126; we also respect output_size.
    mask = (offsets < 126) & (offsets < output_size)

    # Load kernel coefficients (scalar values)
    k0 = tl.load(kernel_ptr + 0)
    k1 = tl.load(kernel_ptr + 1)
    k2 = tl.load(kernel_ptr + 2)

    # Compute the convolution‑like sum for each valid offset
    out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # j = 0
    inp0 = tl.load(input_ptr + offsets + 0, mask=mask, other=0.0)
    out += inp0 * k0

    # j = 1
    inp1 = tl.load(input_ptr + offsets + 1, mask=mask, other=0.0)
    out += inp1 * k1

    # j = 2
    inp2 = tl.load(input_ptr + offsets + 2, mask=mask, other=0.0)
    out += inp2 * k2

    # Store results back to output (only where mask is true)
    tl.store(output_ptr + offsets, out, mask=mask)


def triton_kernel(input, kernel, output, input_size, output_size):
    """
    Triton implementation of the original CUDA kernel.

    Parameters
    ----------
    input : torch.Tensor
        1‑D float32 tensor on CUDA device.
    kernel : torch.Tensor
        1‑D float32 tensor of length 3 on CUDA device.
    output : torch.Tensor
        1‑D float32 tensor on CUDA device where results are written.
    input_size : int
        Size of the input array (kept for API compatibility; not used).
    output_size : int
        Size of the output array; used to bound writes.
    """
    # Basic sanity checks
    assert input.is_cuda and kernel.is_cuda and output.is_cuda, "All tensors must be CUDA tensors"
    assert input.dtype == torch.float32 and kernel.dtype == torch.float32 and output.dtype == torch.float32, \
        "Only float32 tensors are supported"

    # Ensure contiguous layout
    input = input.contiguous()
    kernel = kernel.contiguous()
    output = output.contiguous()

    # Block size used in the original CUDA kernel (126 threads per block).
    # We launch a slightly larger block (128) for better warp alignment.
    BLOCK_SIZE = 128

    # Compute number of blocks following the original launch configuration.
    # The original kernel used blockDim.x = 126.
    num_blocks = (output_size + 126 - 1) // 126
    if num_blocks == 0:
        return

    grid = (num_blocks,)

    # Launch the Triton kernel.
    _triton_kernel_impl[grid](
        input,
        kernel,
        output,
        output_size,
        BLOCK_SIZE,
        num_warps=4,
    )