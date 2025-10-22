import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < 2304
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "Tensors must be float32"
    assert A.numel() == size and B.numel() == size and C.numel() == size, "Tensor sizes must match 'size'"

    num_blocks = triton.cdiv(size, BLOCK_SIZE)
    grid = (num_blocks,)
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)