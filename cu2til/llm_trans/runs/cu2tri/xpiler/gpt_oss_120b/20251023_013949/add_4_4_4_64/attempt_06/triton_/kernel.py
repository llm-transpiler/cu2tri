import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernel: exact functional equivalent of the CUDA kernel above
# ------------------------------------------------------------------
BLOCK_SIZE = 1024  # matches the CUDA __launch_bounds__(1024)

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Parameters
    ----------
    A_ptr, B_ptr, C_ptr : pointers to float32 data (device memory)
    size                : logical vector length (unused inside kernel,
                         kept for API compatibility)
    BLOCK_SIZE          : compile‑time constant = 1024
    """
    pid = tl.program_id(0)                     # block index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread index within the grid

    # Hard‑coded bound from the original CUDA kernel (process only first 4096 elements)
    mask = offs < 4096

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offs, c, mask=mask)


# ------------------------------------------------------------------
# Python wrapper that mimics the original CUDA launch API
# ------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton replacement for:
        void cuda_kernel(float *A, float *B, float *C, int size)

    All tensors must be float32, CUDA‑resident, and contiguous.
    The `size` argument is only used to compute the grid dimensions,
    the kernel still respects the hard‑coded 4096‑element limit.
    """
    # Basic sanity checks (mirrors typical CUDA expectations)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"

    # Compute grid size exactly as the original CUDA launch does
    grid_dim = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_dim,)  # Triton expects a tuple for each launch dimension

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        # num_warps can be tuned; 4 is a safe default for many GPUs
        num_warps=4
    )

    # Synchronize to emulate CUDA's default stream behavior
    torch.cuda.synchronize()