import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel (must be named exactly `_triton_kernel_impl`)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # float* __restrict__ A
    B_ptr,          # float* __restrict__ B
    C_ptr,          # float* __restrict__ T_add (output)
    size,           # int size
    BLOCK_SIZE: tl.constexpr,  # compile‑time block size
):
    """
    Element‑wise addition: C = A + B for `size` elements.
    Out‑of‑range threads are masked out.
    """
    pid = tl.program_id(0)                     # index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # thread offsets within the block

    mask = offsets < size                       # guard against overflow

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    tl.store(C_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper (must be named exactly `triton_kernel`)
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA, contiguous).
    B : torch.Tensor
        Input tensor (float32, CUDA, contiguous).
    C : torch.Tensor
        Output tensor (float32, CUDA, contiguous).
    size : int
        Number of elements to process.
    """
    # ------------------------------------------------------------------
    # Sanity checks (mirroring typical CUDA expectations)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"
    assert size >= 0, "size must be non‑negative"

    # ------------------------------------------------------------------
    # Grid configuration (identical to the CUDA launch configuration)
    # ------------------------------------------------------------------
    BLOCK_SIZE = 1024
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
    )