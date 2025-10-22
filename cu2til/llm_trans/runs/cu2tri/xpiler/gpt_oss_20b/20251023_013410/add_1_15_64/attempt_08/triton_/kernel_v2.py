import triton
import triton.language as tl
import torch

BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(A, B, C, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    a = tl.load(A + offset, mask=mask)
    b = tl.load(B + offset, mask=mask)
    tl.store(C + offset, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    assert A.numel() == size and B.numel() == size and C.numel() == size
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    _triton_kernel_impl[(grid,)](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)