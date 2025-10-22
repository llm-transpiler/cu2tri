import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise addition.
# The kernel name must be exactly `_triton_kernel_impl`.
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # float* __restrict__ A
    B_ptr,               # float* __restrict__ B
    C_ptr,               # float* __restrict__ T_add (output)
    size,                # total number of elements to process
    BLOCK_SIZE: tl.constexpr,   # compile‑time block size (elements per program)
):
    """
    Each Triton program (block) processes BLOCK_SIZE contiguous elements.
    """
    pid = tl.program_id(0)                     # block index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)   # global offsets
    mask = offsets < size                       # guard against OOB

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA `cuda_kernel` signature.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA).
    B : torch.Tensor
        Input tensor B (float32, CUDA).
    C : torch.Tensor
        Output tensor C (float32, CUDA) where the result is stored.
    size : int
        Number of elements to process.
    """
    # Basic sanity checks.
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must reside on the CUDA device.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32.")
    if size < 0:
        raise ValueError("size must be non‑negative.")
    if size > A.numel() or size > B.numel() or size > C.numel():
        raise ValueError("size exceeds tensor element count.")

    # Choose a block size that is a multiple of 32 and works with a power‑of‑two warps count.
    BLOCK_SIZE = 256                     # 8 warps * 32 threads = 256 threads per program
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )   # 1‑D grid

    # Launch the Triton kernel. `num_warps` must be a power of two.
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8  # 8 warps = 256 threads, a power‑of‑two
    )
    # Ensure kernel completion before returning (useful for testing).
    torch.cuda.synchronize()