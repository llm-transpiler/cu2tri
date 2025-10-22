import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: elementwise addition with a hard‑coded bound of 2304
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,                     # *float32
    B_ptr,                     # *float32
    C_ptr,                     # *float32
):
    # compile‑time constants
    BLOCK_SIZE = 1024          # matches CUDA __launch_bounds__(1024)
    MAX_ELEM   = 2304          # hard‑coded bound from the original CUDA kernel

    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global element indices

    # mask for the valid range (identical to the CUDA `if (idx < 2304)`)
    mask = offsets < MAX_ELEM

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper that mimics the original CUDA launch interface
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Launches the Triton kernel with the same signature as the original CUDA kernel.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32 on CUDA.
    size : int
        Logical size of the vectors (used only to compute the grid size;
        the kernel still respects the hard‑coded bound of 2304).
    """
    # Basic sanity checks (mirrors the expectations of the CUDA code)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"

    BLOCK_SIZE = 1024
    # Compute grid dimensions exactly like the CUDA launch configuration
    grid_x = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_x,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C)