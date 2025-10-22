import torch
import triton
import triton.language as tl

# Triton kernel implementation
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # float* __restrict__ A
    B_ptr,          # float* __restrict__ B
    C_ptr,          # float* __restrict__ T_add (output)
    size,           # int size
    BLOCK_SIZE: tl.constexpr,  # compile‑time block size (must be 64)
):
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices within block
    mask = offsets < size                       # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point mirroring the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (must be CUDA, float32, contiguous).
    B : torch.Tensor
        Input tensor B (must be CUDA, float32, contiguous).
    C : torch.Tensor
        Output tensor C (must be CUDA, float32, contiguous).
    size : int
        Number of elements to process.
    """
    # Validate tensors
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must reside on the CUDA device.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of dtype torch.float32.")
    if size < 0:
        raise ValueError("size must be non‑negative.")

    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    BLOCK_SIZE = 64  # matches __launch_bounds__(64) in the CUDA code
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=2,   # 64 threads = 2 warps
    )