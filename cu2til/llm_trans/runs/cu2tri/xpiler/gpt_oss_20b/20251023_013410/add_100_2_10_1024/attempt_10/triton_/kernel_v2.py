import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)

def triton_kernel(A, B, C, size):
    """
    Vector addition kernel: C = A + B

    Parameters
    ----------
    A : torch.Tensor
        Input tensor A, must be a 1D float32 tensor on CUDA device.
    B : torch.Tensor
        Input tensor B, must be a 1D float32 tensor on CUDA device.
    C : torch.Tensor
        Output tensor C, must be a 1D float32 tensor on CUDA device.
    size : int
        Number of elements to process. If None, inferred from A.numel().
    """
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor) or not isinstance(C, torch.Tensor):
        raise TypeError("A, B, and C must be torch.Tensor objects")
    if A.device.type != 'cuda' or B.device.type != 'cuda' or C.device.type != 'cuda':
        raise RuntimeError("A, B, and C must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("A, B, and C must be float32 tensors")
    N = size if size is not None else A.numel()
    if N != B.numel() or N != C.numel():
        raise ValueError("Size mismatch among A, B, and C")
    BLOCK_SIZE = 1024
    grid = triton.cdiv(N, BLOCK_SIZE)
    _triton_kernel_impl[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)