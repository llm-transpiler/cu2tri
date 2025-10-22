import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < 4032
    a = tl.load(A_ptr + offset, mask=mask)
    b = tl.load(B_ptr + offset, mask=mask)
    tl.store(C_ptr + offset, a + b, mask=mask)

def triton_kernel(A, B, C, size):
    BLOCK_SIZE = 1024
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = lambda meta: (num_blocks,)
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)