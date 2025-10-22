```python
import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A, B, C, size):
    BLOCK_SIZE = 1024
    grid = triton.cdiv(size, BLOCK_SIZE)
    _triton_kernel_impl[(,(A, B, C, size,_SIZE=BLOCK)