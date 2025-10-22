import torch
import triton
import triton.language as tl

# Power‑of‑two block size (>= original 960) to satisfy tl.arange requirement
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that adds two vectors element‑wise.
    Mirrors the original CUDA kernel:
        C[i] = A[i] + B[i] for i in [0, size)
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices
    mask = offsets < size                       # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original CUDA launch configuration.
    Parameters
    ----------
    A,, C : torch.Tensor
        1‑D tensors of dtype.float32 residing on the same CUDA device    size : int
        Number of elements process.
    """
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("Only torch.float32 tensors are supported")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise RuntimeError("Tensor sizes are smaller than the requested `size`")

    # Compute grid dimensions – same logic as the CUDA launch pattern
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()