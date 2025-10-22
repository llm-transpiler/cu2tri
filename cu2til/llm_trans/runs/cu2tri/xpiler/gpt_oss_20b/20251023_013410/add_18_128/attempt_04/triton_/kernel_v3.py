import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A, B, C, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    a = tl.load(A + offset, mask=mask, other=0.0)
    b = tl.load(B + offset, mask=mask, other=0.0)
    tl.store(C + offset, a + b, mask=mask)

def triton_kernel(A, B, C, size):
    block_size = 1024
    size = A.shape[0]  # use tensor length to avoid mismatches
    grid = (triton.cdiv(size, block_size),)
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=block_size)