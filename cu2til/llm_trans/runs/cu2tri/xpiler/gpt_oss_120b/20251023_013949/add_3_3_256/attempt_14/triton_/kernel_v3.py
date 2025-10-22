import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernel (must be named exactly `_triton_kernel_impl`)
# ------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float
    B_ptr,          # *float
    C_ptr,          # *float
    BLOCK_SIZE: tl.constexpr,
    NUM_ELEMENTS: tl.constexpr,
):
    """
    Element‑wise addition: C = A + B
    Mirrors the original CUDA kernel's bound check (idx < 2304).
    """
    pid = tl.program_id(0)                     # 1‑D grid index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_ELEMENTS               # enforce hard‑coded limit

    a = tl.load(A_ptr + offsets, mask=mask)    # safe load
    b = tl.load(B_ptr + offsets, mask=mask)    # safe load
    tl.store(C_ptr + offsets, a + b, mask=mask)  # safe store


# ------------------------------------------------------------------
# Wrapper entry point (must be named exactly `triton_kernel`)
# ------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Launches the Triton kernel with the same semantics as the original CUDA kernel.
    Parameters:
        A, B, C : torch tensors on CUDA with dtype torch.float32
        size    : total number of elements (used only for grid size calculation)
    """
    # ------------------- input validation -------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Tensors must be of type torch.float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor storage must be at least `size` elements"

    # ------------------- launch configuration -------------------
    BLOCK_SIZE = 1024
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE   # ceil division
    grid = (num_blocks,)                                 # Triton expects a tuple

    # ------------------- kernel launch -------------------
    _triton_kernel_impl[grid](
        A, B, C,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_ELEMENTS=2304,   # hard‑coded bound from the CUDA kernel
        num_warps=8,
        num_stages=2,
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()