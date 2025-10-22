import triton
import triton.language as tl
import torch

BLOCK_SIZE = 1024
STRIDE = 262144

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    for outer in range(8):
        offset_outer = offsets + outer * stride
        mask = offset_outer < size
        a = tl.load(A_ptr + offset_outer, mask=mask)
        b = tl.load(B_ptr + offset_outer, mask=mask)
        tl.store(C_ptr + offset_outer, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device."
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "All tensors must be float32."
    assert A.numel() == size and B.numel() == size and C.numel() == size, "Tensor sizes must match 'size'."
    grid = 256  # original grid size
    _triton_kernel_impl[grid](A, B, C, size, STRIDE, BLOCK_SIZE=BLOCK_SIZE)