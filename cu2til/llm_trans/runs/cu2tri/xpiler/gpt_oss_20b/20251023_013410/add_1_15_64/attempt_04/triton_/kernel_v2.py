import torch
import triton
import triton.language as tl

# constexpr block size for Triton kernel
BLOCK_SIZE = triton.language.constexpr(960)

@triton.jit
def _triton_kernel_impl(A, B, C, size):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < size
    a_vals = tl.load(A + indices, mask=mask)
    b_vals = tl.load(B + indices, mask=mask)
    tl.store(C + indices, a_vals + b_vals, mask=mask)

def triton_kernel(A, B, C, size):
    size = int(size)
    grid = (triton.cdiv(size, int(BLOCK_SIZE)),)
    _triton_kernel_impl[grid](A, B, C, size)