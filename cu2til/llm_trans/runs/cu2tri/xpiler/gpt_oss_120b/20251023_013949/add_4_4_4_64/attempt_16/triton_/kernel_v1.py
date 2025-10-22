import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementation (named exactly as required)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float32
    B_ptr,          # *float32
    C_ptr,          # *float32
    size,           # i32
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element‑wise addition: C = A + B
    Mirrors the behaviour of the original CUDA kernel.
    """
    pid = tl.program_id(0)                     # grid index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices
    mask = offsets < size                       # bounds check

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function – entry point with the same signature as the CUDA version
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton entry‑point that reproduces the original CUDA kernel semantics.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, contiguous).
    B : torch.Tensor
        Input tensor B (float32, contiguous).
    C : torch.Tensor
        Output tensor C (float32, contiguous).
    size : int
        Number of elements to process.
    """
    # ------------------------------------------------------------------
    # Sanity checks – keep behaviour identical to the CUDA implementation
    # ------------------------------------------------------------------
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "All tensors must be contiguous"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "All tensors must be float32"
    assert size >= 0, "size must be non‑negative"

    # The original CUDA kernel used a fixed block size of 1024 threads
    BLOCK_SIZE = 1024

    # Compute 1‑D grid size (same formula as the CUDA launch)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )