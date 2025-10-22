import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < size
    a = tl.load(A_ptr + indices, mask=mask)
    b = tl.load(B_ptr + indices, mask=mask)
    tl.store(C_ptr + indices, a + b, mask=mask)

def triton_kernel(A, B, C, size):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(size, BLOCK_SIZE),)
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)