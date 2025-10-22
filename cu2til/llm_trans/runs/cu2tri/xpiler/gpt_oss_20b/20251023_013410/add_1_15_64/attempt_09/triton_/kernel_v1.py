import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    A_vals = tl.load(A_ptr + offsets, mask=mask)
    B_vals = tl.load(B_ptr + offsets, mask=mask)
    C_vals = A_vals + B_vals
    tl.store(C_ptr + offsets, C_vals, mask=mask)

def triton_kernel(A, B, C, size):
    BLOCK_SIZE = 960
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)