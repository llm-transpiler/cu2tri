import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, B, C, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton implementation of element‑wise addition.
    Mirrors the behavior of the original CUDA kernel.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices for this block
    mask = offsets < size                       # guard against OOB

    a = tl.load(A, offsets, mask=mask, other=0.0)
    b = tl.load(B, offsets, mask=mask, other=0.0)
    c = a + b

    tl.store(C, offsets, c, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel.
    Has the same signature as the original CUDA entry point:
        cuda_kernel(float *A, float *B, float *C, int size)

    Arguments:
        A (torch.Tensor): Input tensor of shape (size,) and dtype torch.float32 on CUDA.
        B (torch.Tensor): Input tensor of shape (size,) and dtype torch.float32 on CUDA.
        C (torch.Tensor): Output tensor of shape (size,) and dtype torch.float32 on CUDA.
        size (int): Number of elements to process.
    """
    # Basic sanity checks
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32.")
    if A.shape != (size,) or B.shape != (size,) or C.shape != (size,):
        raise RuntimeError("Tensor shapes must match the provided size.")

    BLOCK_SIZE = 960  # matches the CUDA launch bounds
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)