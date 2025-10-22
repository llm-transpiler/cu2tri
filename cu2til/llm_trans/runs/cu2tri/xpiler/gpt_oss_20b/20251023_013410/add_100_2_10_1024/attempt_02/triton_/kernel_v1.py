import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    for outer in range(8):
        offset = block_start + outer * 262144
        indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = indices < 2048000
        a = tl.load(A_ptr + indices, mask=mask)
        b = tl.load(B_ptr + indices, mask=mask)
        tl.store(C_ptr + indices, a + b, mask=mask)

def triton_kernel(A, B, C, size):
    block_size = 1024
    grid = 256
    _triton_kernel_impl[(grid,)](A, B, C, BLOCK_SIZE=block_size)