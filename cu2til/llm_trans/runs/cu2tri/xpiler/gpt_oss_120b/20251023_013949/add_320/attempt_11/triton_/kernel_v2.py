import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                     # block index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size                       # out‑of‑bounds guard
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton equivalent of the original CUDA kernel.
    Parameters match the CUDA signature: (float* A, float* B, float* C, int size).
    """
    # sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor sizes must be at least `size`"

    # ensure contiguous memory for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    BLOCK_SIZE = 256  # power‑of‑two required by tl.arange
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)