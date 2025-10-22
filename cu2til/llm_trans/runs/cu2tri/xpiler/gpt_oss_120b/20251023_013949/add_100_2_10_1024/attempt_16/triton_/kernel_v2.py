import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel performing element‑wise addition:
        C[i] = A[i] + B[i]  for i < size
    Mirrors the behavior of the original CUDA kernel.
    """
    pid = tl.program_id(0)                     # 1‑D grid index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size                       # bounds check
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that launches the Triton kernel.
    Parameters
    ----------
    A, B, C : torch.Tensor
        Input (A, B) and output (C) tensors. Must be CUDA, float32.
    size : int
        Number of elements to process.
    """
    # Sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor storage must be at least `size` elements"

    BLOCK_SIZE = 1024  # matches __launch_bounds__(1024) in the CUDA code

    # Compute 1‑D grid size needed to cover `size` elements
    grid = lambda meta: ((size + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    # Launch the kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

    # Ensure kernel completion before returning
    torch.cuda.synchronize()