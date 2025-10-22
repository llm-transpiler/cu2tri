import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, B, C, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < 2304
    a = tl.load(A + idx, mask=mask)
    b = tl.load(B + idx, mask=mask)
    c = a + b
    tl.store(C + idx, c, mask=mask)

def triton_kernel(A, B, C, size):
    num_blocks = (size + 1023) // 1024
    _triton_kernel_impl[(num_blocks,)](A, B, C, BLOCK_SIZE=1024)