import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024  # matches the CUDA block size (1024 threads)

# ----------------------------------------------------------------------
# Triton kernel
# ----------------------------------------------------------------------
@triton.jit(num_warps=32)  # 32 warps * 32 threads/warp = 1024 threads
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size):
    """
    Element‑wise addition: C = A + B
    Parameters
    ----------
    A_ptr, B_ptr, C_ptr : pointers to float32 buffers (device memory)
    size                : number of elements to process
    """
    pid = tl.program_id(0)                     # block index
    block_start = pid * BLOCK_SIZE              # first element for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # indices handled by this block

    mask = offsets < size                        # guard against out‑of‑bounds
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

# ----------------------------------------------------------------------
# Wrapper entry point (mirrors the original CUDA API)
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton entry point that reproduces the behavior of the original CUDA kernel.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32 on CUDA.
    size : int
        Number of elements to process.
    """
    # ------------------------------------------------------------------
    # Input validation (mirrors typical CUDA expectations)
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32")
    if A.shape != B.shape or A.shape != C.shape:
        raise RuntimeError("All tensors must have the same shape")
    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # ------------------------------------------------------------------
    # Grid configuration (identical to the CUDA launch)
    # ------------------------------------------------------------------
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # 1‑D grid

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](A, B, C, size)