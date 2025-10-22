import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id()
   _start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    N = tl.min(size, 2304)                     # emulate the hard‑coded 2304 bound
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Element‑wise addition of A and B into C, up to a maximum of 2304 elements.
    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors on the same CUDA device.
    size : int
        Expected number of elements in each tensor.
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32
    assert A.numel() == size and B.numel() == size and C.numel() == size

    num_blocks = triton.cdiv(size, BLOCK_SIZE)
    grid = (num_blocks,)
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)