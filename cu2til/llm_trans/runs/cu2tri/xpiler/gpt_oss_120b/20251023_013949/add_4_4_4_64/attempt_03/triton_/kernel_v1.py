import torch
import triton
import triton.language as tl

# Triton kernel that mirrors the original CUDA implementation.
@triton.jit
def _triton_kernel_impl(A, B, C, BLOCK_SIZE: tl.constexpr):
    """
    Element‑wise addition of two float32 vectors with a hard‑coded limit of 4096 elements.
    This reproduces the exact behavior of the CUDA kernel, including the out‑of‑bounds guard.
    """
    pid = tl.program_id(0)                                 # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global element indices
    mask = offsets < 4096                                   # enforce the original 4096‑element bound

    # Load with mask to avoid illegal memory accesses.
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)

    # Store the result only where the mask is true.
    tl.store(C + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original `cuda_kernel` signature.
    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors residing on the same CUDA device.
    size : int
        Logical size used solely to compute the launch grid (identical to the CUDA wrapper).
    """
    # Basic sanity checks.
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, and C must be torch.Tensor objects.")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be on a CUDA device.")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("All tensors must have dtype torch.float32.")
    if A.dim() != 1 or B.dim() != 1 or C.dim() != 1:
        raise ValueError("All tensors must be 1‑D.")

    BLOCK_SIZE = 1024
    # Compute grid size exactly as the original CUDA launch configuration.
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    # Launch the Triton kernel. `num_warps=8` provides ample parallelism on Hopper GPUs.
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE, num_warps=8)