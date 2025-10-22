import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton wrapper replicating the behavior of the original CUDA kernel.
    Parameters:
        A (torch.Tensor): Input tensor of type float32.
        B (torch.Tensor): Input tensor of type float32.
        C (torch.Tensor): Output tensor of type float32.
        size (int): Number of elements to process (should be <= len(A), len(B), len(C)).
    """
    assert isinstance(A, torch.Tensor) and isinstance(B, torch) and isinstance(C, torch.Tensor)
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous()
    assert size <= A.numel() and size <= B.numel() and size <= C.numel()

    BLOCK_SIZE = 1024
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)