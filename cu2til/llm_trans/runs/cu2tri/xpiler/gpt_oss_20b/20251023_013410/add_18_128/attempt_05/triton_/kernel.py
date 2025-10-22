import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < N
    a = tl.load(A_ptr + indices, mask=mask)
    b = tl.load(B_ptr + indices, mask=mask)
    tl.store(C_ptr + indices, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be on CUDA device")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError("All tensors must be of dtype torch.float32")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError("Tensor sizes must be at least 'size'")
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)