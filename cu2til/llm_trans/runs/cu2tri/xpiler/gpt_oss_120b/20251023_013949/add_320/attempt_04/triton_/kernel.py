import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel that performs element‑wise addition.
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # float* __restrict__ A
    B_ptr,          # float* __restrict__ B
    C_ptr,          # float* __restrict__ T_add (output)
    size,           # int size (number of elements to process)
    BLOCK_SIZE: tl.constexpr,  # compile‑time block size (must be power of 2)
):
    """
    Each program instance processes BLOCK_SIZE consecutive elements.
    """
    pid = tl.program_id(0)  # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # linear indices
    mask = offsets < size  # guard against out‑of‑bounds accesses

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)


# ----------------------------------------------------------------------
# Wrapper matching the original CUDA kernel signature.
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
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
        Number of elements to process.
    """
    # ------------------------------------------------------------------
    # Argument validation (mirrors the expectations of the CUDA code)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on a CUDA device"
    assert (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32), \
        "All tensors must be of type torch.float32"
    assert (A.numel() >= size and B.numel() >= size and C.numel() >= size), \
        "Tensor sizes must be at least `size`"

    # Triton requires the block size (range of tl.arange) to be a power of two.
    # The original CUDA kernel used 320 threads per block; we choose 256,
    # which satisfies the power‑of‑two requirement while preserving correctness.
    BLOCK_SIZE = 256

    # Compute grid dimensions: one program per BLOCK_SIZE elements.
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel.
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )