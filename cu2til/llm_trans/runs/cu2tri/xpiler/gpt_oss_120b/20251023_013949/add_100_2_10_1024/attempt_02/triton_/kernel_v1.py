import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, B, C, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes C[i] = A[i] + B[i] for i < size.
    """
    pid = tl.program_id(0  # 1‑D program ID (equivalent blockIdx.x)
    # Offsets for the 1024 threads in this program instance
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask out‑of‑bounds elements
    mask = offsets < size
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel.
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
    # Basic validation
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32")
    # Ensure contiguous memory layout
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    BLOCK_SIZE = 1024
    # Compute grid size (ceil division)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    # Launch kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)