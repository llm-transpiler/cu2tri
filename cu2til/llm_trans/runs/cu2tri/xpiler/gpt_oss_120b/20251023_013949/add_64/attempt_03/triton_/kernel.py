import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(T_add_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton implementation of elementwise addition kernel.
    Mirrors the signature of the original CUDA kernel:
        cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32 on CUDA.
    size : int
        Number of elements to process.
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "All tensors must be float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, "Tensor size insufficient"
    # Ensure contiguous memory layout for optimal performance
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    BLOCK_SIZE = 64
    grid = lambda meta: ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)