import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Configuration constants (mirroring the original CUDA launch parameters)
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024          # threads per block
MAX_ELEMENTS = 4032        # hard‑coded bound from the original CUDA kernel

# ----------------------------------------------------------------------
# Triton kernel implementation
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr, B_ptr, T_add_ptr, size,
    BLOCK_SIZE: tl.constexpr,
    MAX_ELEMENTS: tl.constexpr,
):
    """
    Element‑wise addition kernel.
    Writes `A[i] + B[i]` to `T_add[i]` for indices i satisfying:
        i < size   (logical length supplied by the caller)
        i < MAX_ELEMENTS (original CUDA bound)
    """
    pid = tl.program_id(0)                                 # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices within the block

    # Apply both the logical size bound and the original 4032 element bound
    mask = (offsets < size) & (offsets < MAX_ELEMENTS)

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(T_add_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original `cuda_kernel` signature.

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors residing on the same CUDA device.
    size : int
        Logical length of the vectors (used for grid calculation and bound checking).
    """
    # ------------------------------------------------------------------
    # Input validation (mirrors typical CUDA expectations)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == B.dtype == C.dtype == torch.float32, "All tensors must be float32"
    assert A.dim() == B.dim() == C.dim() == 1, "Only 1‑D tensors are supported"

    # ------------------------------------------------------------------
    # Grid configuration (identical to the original host code)
    # ------------------------------------------------------------------
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    _triton_kernel_impl[(num_blocks,)](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_ELEMENTS=MAX_ELEMENTS,
    )