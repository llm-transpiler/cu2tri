import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing element‑wise addition.
# Uses a power‑of‑two thread count (1024) to satisfy Triton's
# arange requirement, while masking out the extra 64 threads so that
# exactly 960 threads are active per block (as in the original CUDA code).
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *Pointer* to input tensor A (float32)
    B_ptr,          # *Pointer* to input tensor B (float32)
    C_ptr,          # *Pointer* to output tensor C (float32)
    size,           # total number of elements to process (int64)
    BLOCK_SIZE: tl.constexpr,  # compile‑time constant: logical threads per block (960)
):
    pid = tl.program_id(0)                     # block index
    block_start = pid * BLOCK_SIZE              # first element index for this block

    # Triton requires the range length of tl.arange to be a power of two.
    # We therefore use 1024 threads (32 warps) and mask out the surplus.
    RANGE = 1024
    offsets = block_start + tl.arange(0, RANGE)

    # Mask: (1) stay within the logical size, (2) stay within the
    # logical BLOCK_SIZE (960) for each block.
    mask = (offsets < size) & (tl.arange(0, RANGE) < BLOCK_SIZE)

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton entry‑point mirroring the original CUDA kernel signature:
        cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA) with length >= size.
    B : torch.Tensor
        Input tensor (float32, CUDA) with length >= size.
    C : torch.Tensor
        Output tensor (float32, CUDA) with length >= size.
    size : int
        Logical length of the vectors to be added.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must have dtype torch.float32.")

    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    BLOCK_SIZE = 960
    if size <= 0:
        return  # nothing to do

    # ------------------------------------------------------------------
    # Grid configuration (equivalent to CUDA blocks)
    # ------------------------------------------------------------------
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # ------------------------------------------------------------------
    # Launch the kernel.
    # We use 32 warps (1024 threads) because Triton requires the
    # number of warps to be a power of two.
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )