import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, B, C, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes:
        C[i] = A[i] + B[i]   for i in [0, N)
    Mirrors the original CUDA kernel's behavior.
    """
    pid = tl.program_id(0)                     # Block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # Global indices
    mask = offsets < N

    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int) -> None:
    """
    Entry‑point that launches the Triton kernel.
    Parameters match the original CUDA kernel signature:
        A, B, C : torch.float32 tensors on the same CUDA device
        size   : number of elements to process
    """
    # Basic validation (mirrors typical CUDA expectations)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), \
        "size must not exceed tensor lengths"

    BLOCK_SIZE = 1024  # Matches __launch_bounds__(1024) in the CUDA code
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)  # One‑dimensional grid

    # Launch the kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE
    )