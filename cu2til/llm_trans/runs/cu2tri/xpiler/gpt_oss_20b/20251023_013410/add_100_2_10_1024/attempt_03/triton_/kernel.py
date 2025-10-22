import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A, B, C, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    for outer in range(8):
        offset = outer * 262144
        indices = block_start + tl.arange(0, BLOCK_SIZE) + offset
        mask = indices < size
        a = tl.load(A + indices, mask=mask)
        b = tl.load(B + indices, mask=mask)
        tl.store(C + indices, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    assert A.is_cuda and B.is_cuda and C.is_cuda
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32
    BLOCK_SIZE = 1024
    num_blocks = 256
    grid = (num_blocks,)
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)