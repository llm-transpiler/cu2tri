import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, B, C, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size):
    """
    Triton wrapper that mimics the original CUDA kernel.
    A, B, C must be 1‑D float32 tensors on the same CUDA device.
    size is the number of elements to process.
    """
    # Ensure size is a Python int
    size = int(size)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "All tensors must be contiguous"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, "Tensor size insufficient"
    BLOCK_SIZE = 1024
    # Compute grid as a tuple (1D)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    torch.cuda.synchronize()