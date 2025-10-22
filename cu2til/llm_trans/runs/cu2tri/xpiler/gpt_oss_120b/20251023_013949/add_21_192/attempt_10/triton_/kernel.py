import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementation (mirrors the original CUDA kernel)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr,
                        BLOCK_SIZE: tl.constexpr,
                        N: tl.constexpr = 4032):
    """
    Compute C = A + B for indices < N (N = 4032).
    This reproduces the behavior of the original CUDA kernel:
        if (blockIdx.x * 1024 + threadIdx.x) < 4032
            C[idx] = A[idx] + B[idx];
    """
    pid = tl.program_id(0)                     # block index (gridDim.x)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global thread indices
    mask = offsets < N                         # bound check

    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    tl.store(C_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function matching the original CUDA host signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor,
                  B: torch.Tensor,
                  C: torch.Tensor,
                  size: int):
    """
    Entry point that launches the Triton kernel.
    Signature mirrors the original CUDA kernel:
        cuda_kernel(float *A, float *B, float *C, int size)
    """
    # Input validation (mirrors __restrict__ semantics and launch bounds)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"

    BLOCK_SIZE = 1024

    # Compute grid size exactly as in the CUDA host code
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE=BLOCK_SIZE)

    # Ensure kernel completion before returning to Python
    torch.cuda.synchronize()