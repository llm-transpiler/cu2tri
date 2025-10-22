import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel that mirrors the original CUDA implementation.
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr, B_ptr, C_ptr,
    BLOCK_SIZE: tl.constexpr,
    LIMIT: tl.constexpr
):
    """
    Compute C[i] = A[i] + B[i] for i < LIMIT.
    The kernel processes BLOCK_SIZE threads per program (block).
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices
    mask = offsets < LIMIT                     # guard against out‑of‑range

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper that mimics the original CUDA kernel launch interface.
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that launches the Triton kernel.
    Parameters:
        A (torch.Tensor): Input tensor A (float32, CUDA).
        B (torch.Tensor): Input tensor B (float32, CUDA).
        C (torch.Tensor): Output tensor C (float32, CUDA).
        size (int): Number of elements (used only for grid sizing, mirroring the CUDA wrapper).
    """
    # ------------------------------------------------------------------
    # Basic sanity checks (mirroring the expectations of the CUDA code)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"

    # ------------------------------------------------------------------
    # Configuration constants (identical to the original CUDA launch bounds)
    # ------------------------------------------------------------------
    BLOCK_SIZE = 1024          # threads per block
    LIMIT = 2304               # hard‑coded bound from the CUDA kernel

    # ------------------------------------------------------------------
    # Compute grid dimensions exactly as the original CUDA wrapper does.
    # ------------------------------------------------------------------
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Launch the Triton kernel.
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A, B, C,
        BLOCK_SIZE=BLOCK_SIZE,
        LIMIT=LIMIT
    )