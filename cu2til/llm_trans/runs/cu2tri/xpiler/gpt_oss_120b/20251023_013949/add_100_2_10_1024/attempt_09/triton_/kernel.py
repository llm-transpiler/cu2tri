import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes C = A + B for N elements.
    """
    pid = tl.program_id(0)  # 1D grid
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel, matching the original CUDA kernel
    signature: (float* A, float* B, float* C, int size).

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32.
    size : int
        Number of elements to process (must be <= A.numel(), B.numel(), C.numel()).
    """
    # ----- Input validation -----
    if not isinstance(A, torch.Tensor):
        raise TypeError("A must be a torch.Tensor")
    if not isinstance(B, torch.Tensor):
        raise TypeError("B must be a torch.Tensor")
    if not isinstance(C, torch.Tensor):
        raise TypeError("C must be a torch.Tensor")

    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("All tensors must have dtype torch.float32")
    if not A.is_contiguous() or not B.is_contiguous() or not C.is_contiguous():
        raise ValueError("All tensors must be contiguous")
    if A.device != B.device or A.device != C.device:
        raise ValueError("All tensors must reside on the same device")
    if size > A.numel() or size > B.numel() or size > C.numel():
        raise ValueError("size exceeds the number of elements in the provided tensors")

    # ----- Kernel launch configuration -----
    BLOCK_SIZE = 1024  # matches the original CUDA block size
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # 1D grid

    # ----- Launch kernel -----
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

    # Ensure kernel completion before returning
    torch.cuda.synchronize()