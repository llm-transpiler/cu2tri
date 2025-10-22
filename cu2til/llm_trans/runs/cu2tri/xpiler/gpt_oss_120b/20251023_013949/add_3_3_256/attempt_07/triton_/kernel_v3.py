import torch
import triton
import triton.language as tl

# Host‑side constants
BLOCK_SIZE = 1024          # Number of elements processed per program
MAX_IDX = 2304             # Hard‑coded bound from the original CUDA kernel

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, MAX_IDX: tl.constexpr):
    """
    Triton kernel that computes C = A + B for indices < MAX_IDX.
    The `size` argument is kept for API compatibility but is not used inside the kernel.
    """
    pid = tl.program_id(0)                     # 1‑D grid index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # Global indices for this program
    mask = offsets < MAX_IDX                    # Apply the original 2304‑element bound

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel with the same signature as the original CUDA entry point.

    Parameters
    ----------
    A, B, C : torch.Tensor
        Input and output tensors (must be CUDA, float32, contiguous).
    size : int
        Number of elements (used only for grid sizing, matching the original CUDA API).
    """
    # -----------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError("All tensors must be of type torch.float32.")
    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # ---------------------------------------------------------------
    # Compute grid dimensions: one program per BLOCK_SIZE elements
    # ---------------------------------------------------------------
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    if grid[0] <= 0:
        return  # Nothing to do

    # ---------------------------------------------------------------
    # Launch the Triton kernel.
    # Use 32 warps (32 * 32 = 1024 threads) to match the original CUDA block size.
    # ---------------------------------------------------------------
    _triton_kernel_impl[grid](
        A, B, C, size,
        MAX_IDX=MAX_IDX,   # constexpr bound
        num_warps=32
    )