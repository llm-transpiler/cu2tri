import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel_impl(
    A_ptr,
    B_ptr,
    C_ptr,
    size: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise addition of two 1D tensors.
    """
    # Each program instance processes a block of BLOCK_SIZE elements.
    # 1. Get the program ID (pid) for the current instance.
    pid = tl.program_id(axis=0)

    # 2. Calculate the offsets for the elements to be processed by this instance.
    # tl.arange(0, BLOCK_SIZE) creates a range of offsets [0, 1, ..., BLOCK_SIZE-1].
    # These are added to the base offset for the current block (pid * BLOCK_SIZE).
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. Create a mask to prevent out-of-bounds memory access.
    # This is crucial for the last block, which may not be full.
    mask = offsets < size

    # 4. Load the data from global memory.
    # The 'mask' argument ensures that we only load valid data.
    # 'other=0.0' specifies a default value for out-of-bounds elements,
    # preventing potential issues with uninitialized memory, although the
    # result for these elements won't be stored back anyway.
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    # 5. Perform the element-wise addition.
    # This computation is performed on the entire block of data loaded into SRAM.
    c = a + b

    # 6. Store the result back to global memory.
    # The 'mask' argument ensures that we only write to valid memory locations.
    tl.store(C_ptr + offsets, c, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper function for launching the Triton kernel for element-wise addition.

    This function serves as the entry point and has a signature identical to the
    original CUDA wrapper function.

    Args:
        A (torch.Tensor): The first input tensor. Must be a 1D tensor of floats.
        B (torch.Tensor): The second input tensor. Must be a 1D tensor of floats.
        C (torch.Tensor): The output tensor.
        size (int): The number of elements in the tensors.
    """
    # --- Validation ---
    # Ensure all tensors are on the same CUDA device and are contiguous in memory.
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on a CUDA device."
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "All tensors must be contiguous."
    # Verify that the tensor sizes match the provided size parameter.
    assert A.numel() == size and B.numel() == size and C.numel() == size, "Tensor sizes do not match the 'size' parameter."
    # Verify data types.
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "All tensors must be of type float32."

    # --- Grid Configuration ---
    # BLOCK_SIZE is a tunable parameter. 1024 is a good starting point for many GPUs.
    # It determines how many elements each kernel instance will process.
    BLOCK_SIZE = 1024

    # The grid determines the number of kernel instances to launch.
    # We need enough instances to cover all 'size' elements.
    # triton.cdiv(a, b) is a ceiling division, equivalent to (a + b - 1) // b.
    # This is the same logic as `(size + 960 - 1) / 960` in the CUDA code.
    grid = (triton.cdiv(size, BLOCK_SIZE),)

    # --- Kernel Launch ---
    # The kernel is launched with the configured grid.
    # Pointers to tensor data are passed implicitly by passing the tensors themselves.
    # 'size' is passed as a runtime argument.
    # 'BLOCK_SIZE' is passed as a compile-time constant for better performance.
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )