import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Configuration (mirrors the original CUDA launch bounds)
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024          # Threads per block (launch_bounds)
MAX_IDX = 4032             # Upper bound used in the original CUDA kernel

# ----------------------------------------------------------------------
# Triton kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Element‑wise addition: C = A + B
    Only processes indices < MAX_IDX, matching the original CUDA condition.
    """
    pid = tl.program_id(0)                                 # Block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # Thread indices

    mask = offsets < MAX_IDX                               # Boundary mask

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)     # Load A (masked)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)     # Load B (masked)
    c = a + b                                              # Compute sum

    tl.store(C_ptr + offsets, c, mask=mask)               # Write result (masked)

# ----------------------------------------------------------------------
# Wrapper function (identical signature to the original CUDA entry point)
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton wrapper replicating the behavior of the original CUDA kernel.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32) on CUDA device.
    B : torch.Tensor
        Input tensor (float32) on CUDA device.
    C : torch.Tensor
        Output tensor (float32) on CUDA device.
    size : int
        Number of elements (used only for grid sizing, as in the original launch).
    """
    # ------------------------------------------------------------------
    # Basic sanity checks (mirrors typical CUDA expectations)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"

    # ------------------------------------------------------------------
    # Compute grid dimensions based on the provided size (identical to CUDA launch)
    # ------------------------------------------------------------------
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # 1‑D grid

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE=BLOCK_SIZE)

    # Optional: ensure kernel completion before returning to the caller
    torch.cuda.synchronize()