import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < size
    a = tl.load(A_ptr + indices, mask=mask, other=0.0)
    b = tl.load(B_ptr + indices, mask=mask, other=0.0)
    tl.store(C_ptr + indices, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be on CUDA device")
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    block_size = 1024
    grid = (size + block_size - 1) // block_size
    _triton_kernel_impl[(grid,)](A, B, C, size, BLOCK_SIZE=block_size)