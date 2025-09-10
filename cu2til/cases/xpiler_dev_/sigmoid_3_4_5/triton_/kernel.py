import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel_impl(
    A_ptr,
    C_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise sigmoid operation.

    This kernel correctly implements the intended functionality of the original CUDA code,
    which is to apply the sigmoid function element-wise to a 1D tensor. The original
    CUDA kernel had an indexing error (`threadIdx.x` instead of a global index), which
    this implementation corrects by using Triton's standard block-based indexing scheme.
    """
    # 1. Calculate offsets
    # Each program instance (a "block" in CUDA terminology) handles a block of data.
    # `pid` is the program instance ID, similar to `blockIdx.x`.
    pid = tl.program_id(axis=0)
    # `tl.arange` creates a vector of offsets for the threads within this program instance.
    # This is a vectorized operation, replacing a loop or individual thread calculations.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 2. Create a boundary mask
    # Since `n_elements` may not be a multiple of `BLOCK_SIZE`, we need a mask
    # to prevent memory operations from going out of bounds.
    mask = offsets < n_elements

    # 3. Load data
    # Load a block of data from the input tensor `A`. The `mask` ensures that
    # we don't read past the end of the tensor. Invalid lanes are masked off.
    a = tl.load(A_ptr + offsets, mask=mask)

    # 4. Perform computation
    # The operation is `1.0 / (1.0 + exp(-A))`, which is the sigmoid function.
    # Triton's `tl.exp` is used for the exponential function.
    # The computation is applied element-wise to the entire vector `a`.
    output = 1.0 / (1.0 + tl.exp(-a))

    # 5. Store result
    # Store the computed block back to the output tensor `C`. The `mask` ensures
    # that we don't write past the end of the tensor.
    tl.store(C_ptr + offsets, output, mask=mask)


def triton_kernel(A: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper function for the Triton kernel, providing an entry point with a
    signature identical to the original CUDA host function.

    This function launches the Triton kernel to compute the element-wise sigmoid
    of tensor `A` and store the result in tensor `C`.

    Args:
        A (torch.Tensor): The input tensor. Must be a contiguous float32 tensor on a CUDA device.
        C (torch.Tensor): The output tensor. Must be a contiguous float32 tensor on a CUDA device.
        size (int): The number of elements to process.
    """
    # --- Argument Validation ---
    if size == 0:
        return

    assert A.is_cuda and C.is_cuda, "Input and output tensors must be on a CUDA device."
    assert A.dtype == torch.float32 and C.dtype == torch.float32, "Tensors must be of type float32."
    assert A.is_contiguous() and C.is_contiguous(), "Tensors must be contiguous."
    assert A.numel() >= size and C.numel() >= size, "Tensors must have at least 'size' elements."

    # --- Grid Configuration ---
    # We define a 1D grid of programs.
    # BLOCK_SIZE is the number of elements each program will handle.
    # Powers of 2, like 1024, are generally efficient.
    BLOCK_SIZE = 1024

    # `triton.cdiv` is a ceiling division utility function.
    # It calculates the number of blocks needed to cover all `size` elements.
    grid = (triton.cdiv(size, BLOCK_SIZE),)

    # --- Kernel Launch ---
    # The `_triton_kernel_impl` is launched with the configured grid.
    # Pointers to tensors and other arguments are passed.
    # `BLOCK_SIZE` is passed as a `constexpr` for compile-time specialization.
    _triton_kernel_impl[grid](
        A_ptr=A,
        C_ptr=C,
        n_elements=size,
        BLOCK_SIZE=BLOCK_SIZE,
    )