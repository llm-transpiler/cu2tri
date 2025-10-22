import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr,   # *float32
    kernel_ptr,  # *float32 (size 3)
    output_ptr,  # *float32
    output_size: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Original CUDA condition: idx < 126, plus safety against out‑of‑bounds writes
    mask = (offsets < 126) & (offsets < output_size)

    # Load kernel coefficients (scalar loads)
    k0 = tl.load(kernel_ptr + 0)
    k1 = tl.load(kernel_ptr + 1)
    k2 = tl.load(kernel_ptr + 2)

    # Load input values with mask to avoid out‑of‑bounds reads
    in0 = tl.load(input_ptr + offsets + 0, mask=mask, other=0.0)
    in1 = tl.load(input_ptr + offsets + 1, mask=mask, other=0.0)
    in2 = tl.load(input_ptr + offsets + 2, mask=mask, other=0.0)

    # Compute the convolution‑like sum
    out = in0 * k0 + in1 * k1 + in2 * k2

    # Store results only for valid indices
    tl.store(output_ptr + offsets, out, mask=mask)

def triton_kernel(input_tensor, kernel_tensor, output_tensor, input_size, output_size):
    """
    Triton implementation of the original CUDA kernel.
    Parameters match the original CUDA kernel signature:
        input_tensor  : torch.cuda.FloatTensor (device pointer to input)
        kernel_tensor : torch.cuda.FloatTensor (device pointer to kernel, length 3)
        output_tensor : torch.cuda.FloatTensor (device pointer to output)
        input_size    : int (unused, kept for API compatibility)
        output_size   : int (used to compute grid dimensions)
    """
    # Validate inputs
    assert input_tensor.is_cuda and kernel_tensor.is_cuda and output_tensor.is_cuda, \
        "All tensors must be CUDA tensors."
    assert input_tensor.dtype == torch.float32 and kernel_tensor.dtype == torch.float32 \
           and output_tensor.dtype == torch.float32, "All tensors must be float32."
    assert kernel_tensor.numel() == 3, "Kernel tensor must contain exactly 3 elements."

    # Triton block size must be a power of two; choose 128 (>=126)
    BLOCK_SIZE = 128

    # Compute number of program instances (blocks) needed to cover output_size
    num_blocks = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input_tensor,
        kernel_tensor,
        output_tensor,
        output_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )