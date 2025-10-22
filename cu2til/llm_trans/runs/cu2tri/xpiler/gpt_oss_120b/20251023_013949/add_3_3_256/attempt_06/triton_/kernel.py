import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that adds two vectors element‑wise.
    Mirrors the original CUDA kernel logic.
    """
    pid = tl.program_id(axis=0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices within the grid
    mask = offsets < size                            # bounds check (size may be <= 2304)

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)  # load A[i] if in range
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)  # load B[i] if in range
    tl.store(C_ptr + offsets, a + b, mask=mask)         # write C[i] = A[i] + B[i]

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point wrapper that launches the Triton kernel.
    Parameters match the original CUDA kernel signature:
        float* A, float* B, float* C, int size
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"

    BLOCK_SIZE = 1024                     # matches __launch_bounds__(1024)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)  # number of program instances (blocks)

    # Launch the kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

    # Optional: synchronize to make kernel completion explicit
    torch.cuda.synchronize()