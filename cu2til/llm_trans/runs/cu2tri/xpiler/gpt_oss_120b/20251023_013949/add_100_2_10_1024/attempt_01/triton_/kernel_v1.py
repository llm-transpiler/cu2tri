import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N: tl.int32, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id()
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A, B, C, size):
    """
    Triton wrapper that mimics the original CUDA kernel signature.
    Parameters:
        A (torch.Tensor): Input tensor on CUDA device, dtype float32.
        B (torch.Tensor): Input tensor on CUDA device, dtype float32.
        C (torch.Tensor): Output tensor on CUDA device, dtype float32.
        size (int): Number of elements to process.
    """
    # Validation
    assert isinstance(A, torch.Tensor) and A.is_cuda, "A must be a CUDA tensor"
    assert isinstance(B, torch.Tensor) and B.is_cuda, "B must be a CUDA tensor"
    assert isinstance(C, torch.Tensor) and C.is_cuda, "C must be a CUDA tensor"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor sizes must be at least 'size'"

    BLOCK_SIZE = 1024  # matches the original CUDA block size
    grid = (triton.cdiv(size, BLOCK_SIZE),)  # number of program instances (blocks)

    # Launch the Triton kernel; 32 warps => 32*32 = 1024 threads per block
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )