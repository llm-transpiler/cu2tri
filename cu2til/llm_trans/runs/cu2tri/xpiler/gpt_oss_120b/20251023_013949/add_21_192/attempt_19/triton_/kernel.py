import torch
import triton
import triton.language as tl

# Compile-time block size (matches the CUDA launch bounds)
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes:
        C[i] = A[i] + B[i]  for i in [0, size)
    """
    pid = tl.program_id(0)                     # 1‑D grid
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size                       # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A, B, C, size):
    """
    Entry‑point that mimics the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32 on CUDA.
    size : int
        Number of elements to process.
    """
    # Basic sanity checks (mirrors the expectations of the CUDA version)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Only float32 tensors are supported"
    # Compute grid dimensions (same as (size + 1024 - 1) / 1024 in CUDA)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)
    # Ensure completion before returning (optional but mirrors CUDA's synchronous launch)
    torch.cuda.synchronize()