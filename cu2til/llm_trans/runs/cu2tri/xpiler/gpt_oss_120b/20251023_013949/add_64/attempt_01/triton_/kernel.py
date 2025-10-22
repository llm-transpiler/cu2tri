import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementation
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A,          # *float32
    B,          # *float32
    C,          # *float32
    size,       # i32
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element‑wise addition: C[i] = A[i] + B[i] for i in [0, size).
    Each program processes BLOCK_SIZE elements.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread offsets
    mask = offsets < size                       # out‑of‑bounds guard

    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C + offsets, c, mask=mask)


# ----------------------------------------------------------------------
# Wrapper entry point (mirrors the original CUDA kernel signature)
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Launches the Triton kernel to compute C = A + B over `size` elements.

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D contiguous CUDA tensors of dtype torch.float32.
    size : int
        Number of elements to process (must be <= len(A), len(B), len(C)).
    """
    # ------------------------------------------------------------------
    # Argument validation (mirrors typical CUDA host‑side checks)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "Tensors must be contiguous"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), \
        "size exceeds tensor dimensions"

    # ------------------------------------------------------------------
    # Kernel launch configuration
    # ------------------------------------------------------------------
    BLOCK_SIZE = 64  # matches __launch_bounds__(64) in the CUDA version
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # 1‑D grid

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=2,   # small kernel; tweak for performance if needed
        num_stages=2   # default pipeline depth
    )