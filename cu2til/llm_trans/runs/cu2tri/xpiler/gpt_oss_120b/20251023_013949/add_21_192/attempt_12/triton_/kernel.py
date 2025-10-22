import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel (must be named exactly `_triton_kernel_impl`)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_ptr,
                        BLOCK_SIZE: tl.constexpr,
                        BOUND: tl.constexpr):
    """Element‑wise addition with the same bound logic as the CUDA kernel."""
    pid = tl.program_id(0)                                   # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)    # thread indices within block
    mask = offsets < BOUND                                   # emulate the CUDA bound check
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(T_ptr + offsets, a + b, mask=mask)

# ----------------------------------------------------------------------
# Wrapper entry point (must be named `triton_kernel`)
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor,
                  B: torch.Tensor,
                  C: torch.Tensor,
                  size: int):
    """
    Triton implementation of the original CUDA kernel.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32) on CUDA device.
    B : torch.Tensor
        Input tensor (float32) on CUDA device.
    C : torch.Tensor
        Output tensor (float32) on CUDA device.
    size : int
        Logical size of the vectors (used only for grid calculation).
    """
    # ------------------------------------------------------------------
    # Basic sanity checks (mirroring typical CUDA expectations)
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must reside on a CUDA device")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("All tensors must be torch.float32")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise RuntimeError("All tensors must be contiguous")

    BLOCK_SIZE = 1024                     # matches __launch_bounds__(1024)
    # Number of blocks needed to cover `size` elements
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)                  # Triton expects a tuple for the grid

    # Hard‑coded bound from the original CUDA kernel
    BOUND = 4032

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A,                                 # A_ptr
        B,                                 # B_ptr
        C,                                 # T_ptr (output)
        BLOCK_SIZE=BLOCK_SIZE,
        BOUND=BOUND,
        num_warps=4                        # safe default; 1024 threads ≈ 32 warps
    )