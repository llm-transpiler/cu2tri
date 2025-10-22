import triton
import triton.language as tl
import torch

BLOCK_SIZE = 960

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    a = tl.load(A_ptr + offset, mask=mask)
    b = tl.load(B_ptr + offset, mask=mask)
    tl.store(C_ptr + offset, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Element-wise addition of A and B into C.
    A, B, C must be 1D torch.float32 tensors on the same device.
    size is the number of elements to process.
    """
    assert A.numel() == size and B.numel() == size and C.numel() == size
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    _triton_kernel_impl[(grid,)](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)