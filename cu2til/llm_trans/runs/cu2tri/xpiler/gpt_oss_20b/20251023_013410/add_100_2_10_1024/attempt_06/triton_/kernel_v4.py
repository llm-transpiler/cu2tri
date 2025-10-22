import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    block_idx tl.program(axis=0)
    thread_idx = tl.arange(0, BLOCK_SIZE)
    base = block_idx * BLOCK_SIZE + thread_idx
    STRIDE = 262144  # 256 * 1024
    for outer in range(8):
        idx = outer * STRIDE + base
        mask = idx < size
        a = tl.load(A_ptr + idx, mask=mask)
        b = tl.load(B_ptr + idx, mask=mask)
        tl.store(C_ptr + idx, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    grid = 256  # Number of blocks
    BLOCK_SIZE = 1024  # Threads per block
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)