import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, B, C, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for element‑wise addition.
    Mirrors the original CUDA kernel behavior.
    """
    pid = tl.program_id(0)  # block index
    # Global offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask out-of‑bounds indices
    mask = offsets < size

    a = tl.load(A, offsets, mask=mask, other=0.0)
    b = tl.load(B, offsets, mask=mask, other=0.0)
    tl.store(C, offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size):
    """
    Wrapper that launches the Triton kernel.
    Signature matches the original CUDA entry point:
        cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of dtype torch.float32 on CUDA    size : int or torch scalar
        Number of elements to process.
    """
    # Validate device and dtype
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32.")

    # Convert size to a Python int (handles torch scalar inputs)
    size_int = int(size)

    # Ensure tensors contain at least `size` elements
    if A.numel() < size_int or B.numel() < size_int or C.numel() < size_int:
        raise RuntimeError("Tensor sizes must be at least the provided size.")

    # Triton requires the block size (range of tl.arange) to be a power of two.
    # Use 1024 (2^10) which is within the hardware limit of 1024 threads per block.
    BLOCK_SIZE = 1024

    # Compute grid size (number of program instances)
    grid = ((size_int + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size_int, BLOCK_SIZE=BLOCK_SIZE)