import torch
import triton
import triton.language as tl

# Triton kernel implementing the same logic as the CUDA kernel.
# The kernel processes BLOCK_SIZE elements per program (block) and
# only writes results for indices < 126, matching the original CUDA condition.
@triton.jit
def _triton_kernel_impl(
    input_ptr,          # *float32
    kernel_ptr,         # *float32 (size 3)
    output_ptr,         # *float32
    BLOCK_SIZE: tl.constexpr
):
    # Program (block) identifier
    pid = tl.program_id(0)
    # Offsets for this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask to enforce the original idx < 126 condition
    mask = offsets < 126

    # Load kernel coefficients (scalar loads)
    k0 = tl.load(kernel_ptr + 0)
    k1 = tl.load(kernel_ptr + 1)
    k2 = tl.load(kernel_ptr + 2)

    # Load input values with the same mask
    in0 = tl.load(input_ptr + offsets + 0, mask=mask, other=0.0)
    in1 = tl.load(input_ptr + offsets + 1, mask=mask, other=0.0)
    in2 = tl.load(input_ptr + offsets + 2, mask=mask, other=0.0)

    # Compute the convolution-like sum
    out = in0 * k0 + in1 * k1 + in2 * k2

    # Store the result back to output
    tl.store(output_ptr + offsets, out, mask=mask)


def triton_kernel(input_tensor, kernel_tensor, output_tensor, input_size, output_size):
    """
    Wrapper that launches the Triton kernel.
    Parameters match the original CUDA kernel signature:
        input_tensor  : torch.cuda.FloatTensor (device pointer to input)
        kernel_tensor : torch.cuda.FloatTensor (device pointer to kernel, length 3)
        output_tensor : torch.cuda.FloatTensor (device pointer to output)
        input_size    : int (unused, kept for API compatibility)
        output_size   : int (used to compute grid dimensions)
    """
    assert input_tensor.is_cuda and kernel_tensor.is_cuda and output_tensor.is_cuda, \
        "All tensors must be CUDA tensors."
    assert input_tensor.dtype == torch.float32 and kernel_tensor.dtype == torch.float32 \
           and output_tensor.dtype == torch.float32, "All tensors must be float32."

    BLOCK_SIZE = 126  # Must match the CUDA blockDim.x
    # Compute number of programs (blocks) needed to cover output_size
    num_blocks = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the kernel
    _triton_kernel_impl[grid](
        input_tensor,
        kernel_tensor,
        output_tensor,
        BLOCK_SIZE=BLOCK_SIZE,
    )