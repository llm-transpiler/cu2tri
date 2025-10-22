import torch
import triton
import triton.language as tl

# Triton kernel implementation
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float
    B_ptr,          # *float
    C_ptr,          # *float
    BLOCK_SIZE: tl.constexpr,   # compile‑time constant for block size (1024)
    NUM_ELEMENTS: tl.constexpr   # compile‑time constant for bound check (2304)
):
    """
    Element‑wise addition: C = A + B
    Only writes for indices < NUM_ELEMENTS (mirrors the CUDA bounds check).
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # mask for the valid range (same as the CUDA `if (idx < 2304)` guard)
    mask = offsets < NUM_ELEMENTS

    a = tl.load(A_ptr + offsets, mask=mask)    # load A[idx] where mask is True
    b = tl.load(B_ptr + offsets, mask=mask)    # load B[idx] where mask is True
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA launch configuration.
    Arguments:
        A, B, C : torch tensors on CUDA with dtype torch.float32
        size   : total number of elements (used only to compute grid size)
    """
    # Basic sanity checks – mirrors the expectations of the CUDA code
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Tensors must be of type torch.float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor storage must be at least `size` elements"

    BLOCK_SIZE = 1024
    # Compute grid size exactly as in the CUDA wrapper: ceil(size / BLOCK_SIZE)
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel.  The constant NUM_ELEMENTS reproduces the
    # hard‑coded bound check (2304) from the original CUDA kernel.
    _triton_kernel_impl[grid](
        A, B, C,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_ELEMENTS=2304,
        # Optional tuning parameters – can be adjusted for the H800 if needed
        num_warps=8,
        num_stages=2,
    )