import torch
import triton
import triton.language as tl

# Block size matches the CUDA launch bounds
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(
    A,                     # pointer to float32
    B,                     # pointer to float32
    C,                     # pointer to float32 (output)
    size: tl.int32,        # number of elements to process
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that performs element‑wise addition:
        C[i] = A[i] + B[i]   for i < size
    """
    pid = tl.program_id(0)  # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices within block
    mask = offsets < size   # guard against out‑of‑bounds accesses

    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point wrapper that mirrors the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA) with at least `size` elements.
    B : torch.Tensor
        Input tensor (float32, CUDA) with at least `size` elements.
    C : torch.Tensor
        Output tensor (float32, CUDA) with at least `size` elements.
    size : int
        Number of elements to process.
    """
    # ----------------------------------------------------------------------
    # Input validation
    # ----------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor length insufficient for the requested size"

    # Ensure contiguous memory layout for optimal Triton access patterns
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # ----------------------------------------------------------------------
    # Grid configuration (mirrors CUDA's numBlocks calculation)
    # ----------------------------------------------------------------------
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ----------------------------------------------------------------------
    # Kernel launch
    # ----------------------------------------------------------------------
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

    # Synchronize to guarantee completion before returning to the caller
    torch.cuda.synchronize()