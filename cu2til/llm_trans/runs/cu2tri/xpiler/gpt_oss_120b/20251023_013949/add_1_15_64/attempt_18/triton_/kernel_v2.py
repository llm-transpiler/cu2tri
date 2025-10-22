import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, B, C, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """Entry point mirroring the original CUDA kernel signature."""
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be torch.float32")
    BLOCK_SIZE = 960
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )
    torch.cuda.synchronize()