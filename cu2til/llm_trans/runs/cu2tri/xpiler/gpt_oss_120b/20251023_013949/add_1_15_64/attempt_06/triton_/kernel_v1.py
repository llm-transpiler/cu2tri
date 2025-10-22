import torch
import triton
import triton.language as tl

# Triton kernel implementing element‑wise addition.
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float32
    B_ptr,          # *float32
    C_ptr,          # *float32
    size,           # int32
    BLOCK_SIZE: tl.constexpr
):
    # program id == block index
    pid = tl.program_id(0)
    # linear indices for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # mask out‑of‑bounds threads
    mask = offsets < size
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original CUDA host function.
    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors residing on the CUDA device.
    size : int
        Number of elements to process.
    """
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32.")
    # Block size chosen to match the original CUDA launch bounds (960 threads).
    BLOCK_SIZE = 960
    # Compute grid dimensions (ceil division)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE
    )