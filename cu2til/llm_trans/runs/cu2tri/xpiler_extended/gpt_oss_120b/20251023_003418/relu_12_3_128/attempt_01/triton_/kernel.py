import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementation
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024  # matches the CUDA launch bounds

@triton.jit
def _triton_kernel_impl(A_ptr, compute_ptr, size,
                        BLOCK_SIZE: tl.constexpr):
    """
    Compute compute[i] = max(A[i], 0.0) for i < size.
    """
    pid = tl.program_id(0)                     # block index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global offsets
    mask = offs < size

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)      # load with mask
    out = tl.maximum(a, 0.0)                            # max with zero
    tl.store(compute_ptr + offs, out, mask=mask)        # store result


# ----------------------------------------------------------------------
# Wrapper function (entry point) – mirrors the original CUDA signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton entry point that mimics the original CUDA kernel interface.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32) on CUDA device.
    C : torch.Tensor
        Output tensor (float32) on CUDA device.
    size : int
        Number of elements to process.
    """
    # Basic sanity checks
    assert A.is_cuda and C.is_cuda, "Tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and C.dtype == torch.float32, "Tensors must be float32"
    assert A.numel() >= size and C.numel() >= size, "Tensor sizes must be >= `size`"

    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    C = C.contiguous()

    # Grid configuration: one-dimensional grid covering `size` elements
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, C, size, BLOCK_SIZE=BLOCK_SIZE)