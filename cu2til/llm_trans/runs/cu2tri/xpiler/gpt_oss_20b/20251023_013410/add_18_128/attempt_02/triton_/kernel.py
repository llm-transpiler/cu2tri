import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size):
    block_id = tl.program_id(0)
    block_size = 1024
    idx = block_id * block_size + tl.arange(0, block_size)
    mask = idx < size
    a = tl.load(A_ptr + idx, mask=mask)
    b = tl.load(B_ptr + idx, mask=mask)
    tl.store(C_ptr + idx, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    block_size = 1024
    grid = (size + block_size - 1) // block_size
    _triton_kernel_impl[(grid,)](A, B, C, size)