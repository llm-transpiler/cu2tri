import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Host‑side constants
# -------------------------------------------------------------------------
BLOCK_SIZE = 1024          # Elements processed per Triton program (matches CUDA blockDim.x)
MAX_IDX = 2304             # Upper bound used in the original CUDA kernel

# -------------------------------------------------------------------------
# Triton kernel
# -------------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, MAX_IDX: tl.constexpr):
    """
    Triton implementation of the CUDA kernel:
    C[i] = A[i] + B[i] for i < MAX_IDX.
    The `size` argument is kept for API compatibility but is not used inside
    the kernel (the original CUDA kernel also ignored it for the bound check).
    """
    pid = tl.program_id(0)                     # 1‑D grid index
    # Global indices handled by this program (block)
    offsets = pid * 1024 + tl.arange(0, 1024)   # 1024 == BLOCK_SIZE (literal to avoid constexpr issues)
    mask = offsets < MAX_IDX                    # Replicate the original bound check

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


# -------------------------------------------------------------------------
# Wrapper function matching the original CUDA entry point signature
# -------------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Launches the Triton kernel with the same API as the original CUDA kernel.

    Parameters
    ----------
    A, B, C : torch.Tensor
        Input and output tensors (must be CUDA, float32, contiguous).
    size : int
        Number of elements (used only for grid sizing, mirroring the CUDA launch).
    """
    # ------------------------------
    # Input validation
    # ------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError("All tensors must be of type torch.float32.")
    # Ensure contiguous memory layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # ------------------------------
    # Grid configuration
    # ------------------------------
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)  # tuple required by Triton
    if grid[0] <= 0:
        return  # Nothing to do

    # ------------------------------
    # Kernel launch
    # ------------------------------
    _triton_kernel_impl[grid](
        A, B, C, size,
        MAX_IDX=MAX_IDX,   # constexpr bound
        num_warps=32       # 32 warps = 1024 threads, matching the original block size
    )