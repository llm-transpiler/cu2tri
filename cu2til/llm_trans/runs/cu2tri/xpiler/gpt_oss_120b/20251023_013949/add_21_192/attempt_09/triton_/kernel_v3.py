import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Configuration constants (mirroring the original CUDA launch parameters)
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024          # threads per block
MAX_ELEMENTS = 4032        # hard‑coded bound from the CUDA kernel

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
    Computes T_add[i] = A[i] + B[i] for i < min(size, MAX_ELEMENTS).
    """
    pid = tl.program_id(0)                                 # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # thread indices

    # Apply both the logical size bound and the hard‑coded 4032 bound
    mask = (offsets < size) & (offsets < MAX_ELEMENTS)

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(T_add_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size):
    """
    Entry‑point that mimics the original `cuda_kernel` signature.

    Parameters
    ----------
    A, B, C : torch.Tensor
        Input and output tensors. Must be on the same CUDA device,
        have dtype torch.float32, be contiguous, and contain at least `size`
        elements. They may have any shape; they are flattened internally.
    size : int or torch scalar
        Logical length of the vectors.
    """
    # ------------------------------------------------------------------
    # Input validation (mirrors typical CUDA expectations)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == B.dtype == C.dtype == torch.float32, "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "Tensors must be contiguous"

    # Convert size to a Python int (handles torch scalar inputs)
    size = int(size)

    # Flatten tensors to 1‑D views (no copy thanks to contiguity)
    A_flat = A.view(-1)
    B_flat = B.view(-1)
    C_flat = C.view(-1)

    # Verify that the buffers are large enough for the requested logical size
    assert A_flat.numel() >= size and B_flat.numel() >= size and C_flat.numel() >= size, \
        "Tensor size insufficient for the requested `size`"

    # ------------------------------------------------------------------
    # Grid configuration (identical to the original host code)
    # ------------------------------------------------------------------
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    _triton_kernel_impl[(num_blocks,)](
        A_flat,
        B_flat,
        C_flat,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_ELEMENTS=MAX_ELEMENTS,
    )