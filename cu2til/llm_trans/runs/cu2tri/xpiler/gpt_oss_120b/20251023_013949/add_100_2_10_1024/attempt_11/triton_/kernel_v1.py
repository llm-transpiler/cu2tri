import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that adds two float32 vectors element‑wise.
    Parameters
    ----------
    A_ptr, B_ptr, C_ptr : pointers to float32 buffers (device memory)
    N : total number of elements to process
    BLOCK_SIZE : compile‑time constant, number of threads per program instance
    """
    pid = tl.program_id(0)                     # 1‑D grid
    # Compute absolute indices for this thread
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Guard against out‑of‑bounds accesses
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A, B, C : torch.cuda.FloatTensor of at least `size` elements
    size    : number of elements to add
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor storage insufficient for the requested size"

    BLOCK_SIZE = 1024  # matches __launch_bounds__(1024) in the CUDA version
    # Compute grid size: enough blocks to cover `size` elements
    grid = (triton.cdiv(size, BLOCK_SIZE),)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)