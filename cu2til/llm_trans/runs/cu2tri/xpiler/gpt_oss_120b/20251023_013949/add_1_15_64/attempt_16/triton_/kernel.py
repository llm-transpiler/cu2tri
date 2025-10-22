import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, B, C, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton implementation of elementwise addition:
    C[i] = A[i] + B[i] for i in [0, size)
    """
    # Validate inputs
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be on CUDA device")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise ValueError("All tensors must be torch.float32")
    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    # Block size must be <= num_warps * 32 (warp size)
    BLOCK_SIZE = 1024  # 32 warps * 32 threads per warp
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,  # power of two as required by Triton
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()