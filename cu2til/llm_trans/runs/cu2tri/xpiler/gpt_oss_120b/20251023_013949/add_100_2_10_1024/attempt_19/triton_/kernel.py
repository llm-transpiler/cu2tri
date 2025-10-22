import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, B, C, size,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Linear offset for each thread within the grid
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Guard against out‑of‑bounds accesses
    mask = offsets < size
    a = tl.load(A + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)
    tl.store(C + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton wrapper that reproduces the behavior of the original CUDA kernel.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32) on CUDA device.
    B : torch.Tensor
        Input tensor (float32) on CUDA device.
    C : torch.Tensor
        Output tensor (float32) on CUDA device.
    size : int
        Number of elements to process (must be <= A.numel()).
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "Only float32 tensors are supported"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, "Tensor sizes must be at least `size`"
    BLOCK_SIZE = 1024 # matches the CUDA launch bounds
    # Compute grid dimension: enough blocks to cover `size` elements
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)