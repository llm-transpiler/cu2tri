import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes element‑wise addition:
        C[i] = A[i] + B[i]   for i in [0, size)
    BLOCK_SIZE is a compile‑time constant (default 960) matching the
    original CUDA launch bounds.
    """
    pid = tl.program_id(0)                     # block index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread offsets within the block
    mask = offs < size                         # guard out‑of‑bounds threads

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    tl.store(C_ptr + offs, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original CUDA kernel launch.
    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D tensors of dtype torch.float32 residing on the GPU.
    size : int
        Number of elements to process.
    """
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("Only torch.float32 tensors are supported.")
    if size < 0:
        raise ValueError("size must be non‑negative.")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError("Tensor length is smaller than the requested size.")

    BLOCK_SIZE = 960  # matches __launch_bounds__(960) in the CUDA code
    # Compute grid dimensions (one‑dimensional grid)
    grid = lambda meta: ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )