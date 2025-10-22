import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: elementwise max with zero (mirrors the original CUDA kernel)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A, compute, size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    compute[i] = max(A[i], 0.0)  for i in [0, size)
    """
    pid = tl.program_id(0)  # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global thread IDs
    mask = offsets < size  # bound check

    # Load, compute max with zero, and store (masked)
    a_val = tl.load(A + offsets, mask=mask, other=0.0)
    out_val = tl.maximum(a_val, 0.0)
    tl.store(compute + offsets, out_val, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function: entry point with the same signature as the CUDA wrapper
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that mimics the original `cuda_kernel` signature:
        void cuda_kernel(float *A, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA, contiguous).
    C : torch.Tensor
        Output tensor (float32, CUDA, contiguous). Must have at least `size` elements.
    size : int
        Number of elements to process.
    """
    # ------------------------------------------------------------------
    # Input validation (mirrors expectations of the original CUDA code)
    # ------------------------------------------------------------------
    assert isinstance(A, torch.Tensor) and isinstance(C, torch.Tensor), "A and C must be torch tensors"
    assert A.is_cuda and C.is_cuda, "A and C must be CUDA tensors"
    assert A.dtype == torch.float32 and C.dtype == torch.float32, "Only float32 tensors are supported"
    assert A.is_contiguous() and C.is_contiguous(), "Tensors must be contiguous"
    assert A.numel() >= size and C.numel() >= size, "Tensors must contain at least `size` elements"

    # ------------------------------------------------------------------
    # Kernel launch configuration (identical to the CUDA wrapper)
    # ------------------------------------------------------------------
    BLOCK_SIZE = 1024
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)  # 1D grid

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](A, C, size, BLOCK_SIZE)