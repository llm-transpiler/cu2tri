import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, block_size: tl.constexpr):
    idx = tl.program_id(0) * block_size + tl.thread_idx(0)
    if idx < 2304:
        a = tl.load(A_ptr + idx)
        b = tl.load(B_ptr + idx)
        tl.store(C_ptr + idx, a + b)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    block_size = 1024
    grid = (size + block_size - 1) // block_size
    _triton_kernel_impl[(grid,)](A, B, C, block_size=block_size)