import torch
import triton
import triton.language as tl

# Number of threads per program (equivalent to CUDA blockDim.x)
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton implementation of element‑wise addition:
        C[i] = A[i] + B[i]   for i < size
    """
    pid = tl.program_id(0)                     # program (block) index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices
    mask = offsets < size                       # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that mirrors the original CUDA kernel signature:
        triton_kernel(float* A, float* B, float* C, int size)

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D CUDA tensors of dtype torch.float32.
    size : int
        Number of elements to process.
    """
    # Basic validation (mirrors the expectations of the original CUDA code)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor storage must be at least `size` elements"

    # Compute grid size (equivalent to CUDA's numBlocks)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel. 32 warps => 1024 threads per program, matching __launch_bounds__(1024)
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32
    )