import torch
import triton
import triton.language as tl

# Triton requires the range length of tl.arange to be a power of two.
# 256 is a convenient choice that still covers the original 320‑element case
# with a simple tail mask.
BLOCK_SIZE = 256

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel performing element‑wise addition:
        C[i] = A[i] + B[i]   for i in [0, N)
    Mirrors the behavior of the original CUDA kernel.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # absolute indices
    mask = offsets < N                          # guard for tail elements

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point wrapper matching the original `cuda_kernel` signature.

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors residing on the same CUDA device.
    size : int
        Number of elements to process (must be ≤ len(A), len(B), len(C)).
    """
    # Basic validation (mirrors expectations of the CUDA version)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Only float32 tensors are supported"
    assert A.ndim == B.ndim == C.ndim == 1, "Only 1‑D tensors are supported"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), \
        "size exceeds tensor length"

    # Compute grid: one program (block) per BLOCK_SIZE elements
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)