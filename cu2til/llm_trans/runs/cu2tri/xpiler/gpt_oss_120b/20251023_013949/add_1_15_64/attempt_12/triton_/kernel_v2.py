import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, B, C, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel implementing element‑wise addition.
    Mirrors the original CUDA kernel behavior.
    """
    pid = tl.program_id(0)  # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices for this block
    mask = offsets < size  # guard against out‑of‑bounds

    a = tl.load(A, offsets, mask=mask, other=0.0)
    b = tl.load(B, offsets, mask=mask, other=0.0)
    tl.store(C, offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size):
    """
    Wrapper that launches the Triton kernel.
    Signature matches the original CUDA entry point:
        cuda_kernel(float *A, float *B, float *C, int    Parameters
    ----------
    A : torch.Tensor
        Input tensor of dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of dtype torch.float32 on CUDA.
    size : int or torch scalar
        Number of elements to process.
    """
    # Validate device and dtype
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32.")

    # Convert size to a Python int (handles torch scalar inputs)
    size_int = int(size)

    # Ensure tensors contain at least `size_int` elements
    if A.numel() < size_int or B.numel() < size_int or C.numel() < size_int:
        raise RuntimeError("Tensor sizes must be at least the provided size.")

    BLOCK_SIZE = 960  # matches CUDA launch bounds
    grid = ((size_int + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size_int, BLOCK_SIZE=BLOCK_SIZE)