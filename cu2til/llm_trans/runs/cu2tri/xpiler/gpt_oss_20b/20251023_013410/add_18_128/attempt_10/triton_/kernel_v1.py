import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A, B, C, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < size
    a = tl.load(A + indices, mask=mask, other=0.0)
    b = tl.load(B + indices, mask=mask, other=0.0)
    c = a + b
    tl.store(C + indices, c, mask=mask)

def triton_kernel(A, B, C, size):
    """
    Wrapper for Triton kernel. A, B, C are torch tensors on CUDA device.
    size is the number of elements to process.
    """
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be on CUDA device")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError("All tensors must be of type torch.float32")
    BLOCK_SIZE = 1024
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    _triton_kernel_impl[(grid,)](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)