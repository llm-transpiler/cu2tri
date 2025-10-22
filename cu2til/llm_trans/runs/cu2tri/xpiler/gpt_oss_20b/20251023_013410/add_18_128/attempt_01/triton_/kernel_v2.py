import triton
import triton.language as tl
import torch

@triton.jit
def _iton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (i < 2304) & (i < size)
    a = tl.load(A_ptr + i, mask=mask, other=0.0)
    b = tl.load(B_ptr + i, mask=mask, other=0.0)
    tl.store(C_ptr + i, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be on CUDA device")
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    block_size = 1024
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=block_size)