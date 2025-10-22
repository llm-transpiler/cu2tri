# Triton implementation of the CUDA kernel
import torch
import triton
import triton.language as tl

# Compile‑time block size (matches the CUDA launch bounds)
BLOCK_SIZE = 960

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that adds two vectors element‑wise.
    Mirrors the semantics of the original CUDA kernel:
        C[i] = A[i] + B[i] for i in [0, size)
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices
    mask = offsets < size                       # guard against out‑of‑bounds

    # Load values (use 0.0 for masked‑off elements – they will never be written)
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    # Store the result
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original CUDA launch configuration.
    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D tensors of type torch.float32 residing on the same CUDA device.
    size : int
        Number of elements to process """
    # Basic sanity checks
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("Only torch.float32 tensors are supported")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise RuntimeError("Tensor sizes are smaller than the requested `size`")

    # Compute grid dimensions – identical to the CUDA launch pattern
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )