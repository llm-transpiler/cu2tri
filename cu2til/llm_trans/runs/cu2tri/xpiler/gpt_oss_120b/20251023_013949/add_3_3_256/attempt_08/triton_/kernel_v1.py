import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise addition with the same bounds as the original CUDA kernel.
@triton.jit
def _triton_kernel_impl(A, B, C):
    # Compile‑time constants
    BLOCK_SIZE = 1024
    BOUND = 2304

    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global element indices for this block
    mask = offsets < BOUND

    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D tensors of dtype torch.float32 residing on the same CUDA device.
    size : int
        Logical size of the vectors (kept for API compatibility; the kernel
        still respects the hard‑coded bound of 2304 elements as in the CUDA
        reference).
    """
    # Basic sanity checks – these mirror the expectations of the original CUDA launch.
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"

    BLOCK_SIZE = 1024
    # Compute the number of blocks exactly as the CUDA host code does.
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel.
    _triton_kernel_impl[grid](A, B, C)