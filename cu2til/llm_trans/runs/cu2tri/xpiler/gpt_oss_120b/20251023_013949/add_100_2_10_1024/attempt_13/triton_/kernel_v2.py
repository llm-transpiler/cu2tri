import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    A_ptr,
    B_ptr,
    C_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < size
    a = tl.load(A_ptr + offs, mask=mask)
    b = tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton wrapper mimicking the original CUDA kernel.
    Parameters:
        A (torch.Tensor): Input tensor A (float32, CUDA).
        B (torch.Tensor): Input tensor B (float32, CUDA).
        C (torch.Tensor): Output tensor C (float32, CUDA).
        size (int): Number of elements to process.
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "All tensors must be float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, "Tensor size mismatch"

    BLOCK_SIZE = 1024  # Matches the original CUDA block size
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE  # Number of program instances

    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,  # 1024 threads = 32 warps
    )