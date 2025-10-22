import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton implementation of elementwise addition.
    Parameters:
        A (torch.Tensor): Input tensor of shape (size,) and dtype torch.float32.
        B (torch.Tensor): Input tensor of shape (size,) and dtype torch.float32.
        C (torch.Tensor): Output tensor of shape (size,) and dtype torch.float32.
        size (int): Number of elements to process.
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "Tensors must be float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, "Tensor size insufficient"

    BLOCK_SIZE = 1024
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8  # tuned for H800
    )