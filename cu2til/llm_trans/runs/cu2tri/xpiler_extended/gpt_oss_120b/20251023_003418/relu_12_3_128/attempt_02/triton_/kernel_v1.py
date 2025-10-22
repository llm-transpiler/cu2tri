import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: elementwise max with zero (identical to the CUDA version)
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024          # matches the CUDA block size
MAX_SIZE   = 4608          # hard‑coded bound used in the original CUDA kernel

@triton.jit
def _triton_kernel_impl(A, compute):
    """
    Triton kernel that mirrors the CUDA implementation:
        compute[i] = max(A[i], 0.0)   if i < 4608
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # global thread IDs
    mask = offsets < MAX_SIZE                   # bound check (same as CUDA)

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
        Output tensor (float32, CUDA, contiguous). Must be at least `size` elements.
    size : int
        Logical size used only to compute the grid configuration.
    """
    # Basic sanity checks (mirrors expectations of the CUDA code)
    assert A.is_cuda and C.is_cuda, "A and C must be CUDA tensors"
    assert A.dtype == torch.float32 and C.dtype == torch.float32, "Only float32 supported"
    assert A.is_contiguous() and C.is_contiguous(), "Tensors must be contiguous"

    # Compute grid dimensions exactly as in the CUDA wrapper
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, C)