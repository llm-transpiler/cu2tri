import torch
import triton
import triton.language as tl

# Block size matches the CUDA launch bounds (1024 threads per block)
BLOCK_SIZE = 1024

@triton.jit(num_warps=32)  # 1024 threads = 32 warps (32 threads per warp)
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that performs element‑wise addition:
        C[i] = A[i] + B[i]   for i < size
    """
    pid = tl.program_id(0)                     # 1‑D grid
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size                       # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point wrapper that mimics the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA, contiguous)
    B : torch.Tensor
        Input tensor B (float32, CUDA, contiguous)
    C : torch.Tensor
        Output tensor C (float32, CUDA, contiguous)
    size : int
        Number of elements to process
    """
    # Basic validation (mirrors expectations of the CUDA version)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Only float32 tensors are supported"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"

    # Compute grid dimensions (identical to CUDA's (size + 1024 - 1) / 1024)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

    # Ensure kernel completion before returning (optional but mirrors CUDA sync semantics)
    torch.cuda.synchronize()