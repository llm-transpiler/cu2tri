import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size):
    # Each program processes a block of 1024 elements
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    # Bound check: only compute for indices < 2304 (mirrors original CUDA)
    mask = offsets < 2304
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton equivalent of the original CUDA kernel.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA)
    B : torch.Tensor
        Input tensor B (float32, CUDA)
    C : torch.Tensor
        Output tensor C (float32, CUDA)
    size : int
        Number of elements (used only for grid sizing, matches original API)
    """
    # Validate inputs
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise ValueError("All tensors must be of type torch.float32")
    # Ensure contiguous memory layout
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    # Compute grid size (mirrors original CUDA launch configuration)
    grid = (size + 1024 - 1) // 1024
    if grid <= 0:
        return  # nothing to do
    _triton_kernel_impl[grid](A, B, C, size)