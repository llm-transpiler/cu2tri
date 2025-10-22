import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementation (mirrors the CUDA kernel)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # float* __restrict__ A
    B_ptr,          # float* __restrict__ B
    T_add_ptr,      # float* __restrict__ T_add (output)
    size,           # int size (unused, kept for API compatibility)
    BLOCK_SIZE: tl.constexpr,   # compile‑time block size (1024)
):
    """
    Element‑wise addition: T_add[i] = A[i] + B[i] for i < 4096.
    The guard mirrors the original CUDA kernel's constant‑bound check.
    """
    pid = tl.program_id(0)                     # blockIdx.x
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # threadIdx.x + blockIdx.x*BLOCK_SIZE

    # Guard: only compute for the first 4096 elements (exact CUDA behavior)
    mask = offsets < 4096

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(T_add_ptr + offsets, c, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper that mimics the original `cuda_kernel` signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that launches the Triton kernel.
    Parameters match the original CUDA kernel:
        - A, B, C: torch.cuda.FloatTensor (float32) with at least `size` elements
        - size: logical length of the vectors (used only for grid calculation)
    """
    # ------------------------------------------------------------------
    # Argument validation (mirrors typical CUDA expectations)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Input tensors must contain at least `size` elements"

    # ------------------------------------------------------------------
    # Grid configuration (identical to the CUDA launch)
    # ------------------------------------------------------------------
    BLOCK_SIZE = 1024
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE   # ceil division

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE
    )