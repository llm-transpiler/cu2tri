import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton equivalent of the CUDA kernel `cuda_kernel`.
    Performs elementwise addition: C[i] = A[i] + B[i] for i < size.
    """
    # Validate inputs
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise ValueError("All tensors must be of type torch.float32")
    # Ensure contiguous layout
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    BLOCK_SIZE = 1024
    # Compute grid dimensions
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8  # Adjust for target GPU; 8 warps works well on Hopper
    )