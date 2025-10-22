import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Triton kernel: element‑wise addition with the same semantics as the CUDA
# implementation.  The kernel processes 1024 elements per program (block) and
# respects the original hard‑coded bound of 2304 elements.
# -------------------------------------------------------------------------
BLOCK_SIZE = 1024          # Number of elements handled per program
MAX_IDX = 2304             # Upper bound used in the original CUDA kernel

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size):
    """
    Triton kernel that computes C = A + B for indices < MAX_IDX.
    Parameters
    ----------
    A_ptr, B_ptr, C_ptr : tl.pointer
        Pointers to the input/output buffers (float32).
    size : int
        Original size argument (unused in the computation, kept for API compatibility).
    """
    pid = tl.program_id(0)                     # 1‑D grid index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # Global indices for this program
    mask = offsets < MAX_IDX                    # Apply the original 2304‑element bound

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel with the same signature as the
    original CUDA entry point.

    Parameters
    ----------
    A, B, C : torch.Tensor
        Input and output tensors (must be CUDA, float32, contiguous).
    size : int
        Number of elements (used only for grid sizing, matching the CUDA API).
    """
    # -----------------------------------------------------------------
    # Input validation (mirrors expectations of the original CUDA code)
    # -----------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError("All tensors must be of type torch.float32.")
    # Ensure contiguous memory layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # ---------------------------------------------------------------
    # Compute grid dimensions: one program per 1024‑element chunk
    # ---------------------------------------------------------------
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)  # tuple, not int
    if grid[0] <= 0:
        return  # Nothing to do

    # ---------------------------------------------------------------
    # Launch the Triton kernel.
    # We request 32 warps (32 * 32 = 1024 threads) to match the CUDA block size.
    # ---------------------------------------------------------------
    _triton_kernel_impl[grid](A, B, C, size, num_warps=32)