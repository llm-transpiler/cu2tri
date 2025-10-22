import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementation
# Mirrors the CUDA kernel _cuda_kernel_impl
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # float* __restrict__ A
    B_ptr,          # float* __restrict__ B
    C_ptr,          # float* __restrict__ T_add (output)
    size,           # int size (passed from wrapper)
    BLOCK_SIZE: tl.constexpr,  # compile‑time block size (1024)
):
    """
    Perform element‑wise addition: C[i] = A[i] + B[i]
    for i < min(size, 2304).  The constant 2304 reproduces the exact
    bound check present in the original CUDA kernel.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices

    # Apply the same two‑fold bound check as the CUDA version
    mask = (offsets < size) & (offsets < 2304)

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    tl.store(C_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Public wrapper matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that replicates the behaviour of the original `cuda_kernel`
    function.

    Parameters
    ----------
    A : torch.Tensor
 Input of shapesize,) on CUDA device, dtype.float32    B : torch.Tensor
        Input tensor of shape (size,) on CUDA device, dtype torch.float32.
    C : torch.Tensor
        Output tensor of shape (size,) on CUDA device, dtype torch.float32.
    size : int
        Logical size of the vectors (used only for grid calculation and
        bound checking inside the kernel).
    """
    # ------------------------------------------------------------------
    # Basic sanity checks – mirrors the expectations of the CUDA code
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor storage must be at least `size` elements"

    BLOCK_SIZE = 1024  # matches __launch_bounds__(1024) in CUDA

    # Compute grid size exactly like the CUDA launch configuration
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
    )