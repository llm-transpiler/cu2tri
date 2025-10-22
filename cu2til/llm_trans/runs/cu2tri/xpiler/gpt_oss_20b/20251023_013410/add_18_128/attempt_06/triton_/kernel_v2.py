import triton
import triton.language as tl
import torch


@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N):
    pid = tl.program_id(0)
    block_size = 1024
    offset = pid * block_size
    indices = offset + tl.arange(0, block_size)
    mask = indices < N
    a = tl.load(A_ptr + indices, mask=mask)
    b = tl.load(B_ptr + indices, mask=mask)
    tl.store(C_ptr + indices, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Wrapper that launches the Triton kernel.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32).
    B : torch.Tensor
        Input tensor B (float32).
    C : torch.Tensor
        Output tensor C (float32). Must be preallocated with the same size as A and B.
    size : int
        Number of elements to process.
    """
    # Ensure tensors are on CUDA and of type float32
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32, device='cuda')
    if not isinstance(B, torch.Tensor):
        B = torch.tensor(B, dtype=torch.float32, device='cuda')
    if not isinstance(C, torch.Tensor):
        C = torch.tensor(C, dtype=torch.float32, device='cuda')

    if A.device != B.device or A.device != C.device:
        raise ValueError("All tensors must be on the same device")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("All tensors must be of dtype float32")
    if A.numel() != B.numel() or A.numel() != C.numel():
        raise ValueError("All tensors must have the same number of elements")
    if size > A.numel():
        raise ValueError("size must not exceed the number of elements in the tensors")

    BLOCK_SIZE = 1024
    grid = triton.cdiv(size, BLOCK_SIZE)
    _triton_kernel_impl[(grid,)](A, B, C, size)