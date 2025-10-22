import triton
import triton.language as tl
import torch

BLOCK_SIZE = 960

@triton.jit
def _triton_kernel_impl(A, B, C, size):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < size
    C[indices] = tl.load(A + indices, mask=mask) + tl.load(B + indices, mask=mask)

def triton_kernel(A, B, C, size):
    grid = (triton.cdiv(size, BLOCK_SIZE),)
    _triton_kernel_impl[grid](A, B, C, size)