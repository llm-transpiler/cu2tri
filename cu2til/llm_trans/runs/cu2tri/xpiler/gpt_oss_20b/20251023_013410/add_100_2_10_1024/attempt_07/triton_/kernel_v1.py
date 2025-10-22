import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    for i in range(pid, N, BLOCK_SIZE):
        a = tl.load(A_ptr + i)
        b = tl.load(B_ptr + i)
        tl.store(C_ptr + i, a + b)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    BLOCK_SIZE = 1024
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    _triton_kernel_impl[(grid,)](A, B, C, size BLOCK=BLOCK_SIZE)