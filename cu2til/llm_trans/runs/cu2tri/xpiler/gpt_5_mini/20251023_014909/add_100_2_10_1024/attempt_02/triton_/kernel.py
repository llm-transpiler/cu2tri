import torch
import triton
import triton.language as tl

# Triton kernel implementation (must be named exactly as requested)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK: tl.constexpr):
    """
    A_ptr, B_ptr, C_ptr: 1D device pointers (torch tensors passed directly)
    size: total number of valid elements (int)
    BLOCK: number of "threads" per block (constexpr, expected 1024)
    This kernel mirrors the CUDA kernel's indexing pattern:
      for outer in range(8):
        index = outer * 262144 + blockIdx.x * 1024 + threadIdx.x
        if index < 2048000: C[index] = A[index] + B[index]
    We use `size` as the bound check (safer / more general). To exactly
    reproduce the CUDA constant behavior, pass size=2048000.
    """
    pid = tl.program_id(0)                       # analogous to blockIdx.x
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # threadIdx range per program

    # stride between successive "outer" iterations in the original CUDA code
    STRIDE = 262144  # 256 * 1024
    for k in range(8):
        idx = offs + k * STRIDE
        mask = idx < size
        a = tl.load(A_ptr + idx, mask=mask, other=0.0)
        b = tl.load(B_ptr + idx, mask=mask, other=0.0)
        tl.store(C_ptr + idx, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Host-side wrapper that configures and launches the Triton kernel.

    Signature mirrors the original host entry:
      cuda_kernel(float *A, float *B, float *C, int size)

    Parameters:
      A, B, C : torch.Tensor
        1D (or viewable as 1D) CUDA tensors with dtype=torch.float32.
      size : int
        Number of valid elements to operate on (use 2048000 to match the CUDA
        constant in the original kernel).
    """
    # Basic argument validation to keep behavior consistent and safe
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this Triton kernel.")
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor instances.")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise TypeError("A, B, C must be CUDA tensors (on GPU).")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must have dtype torch.float32.")
    # Flatten to 1D views (the original kernel treats buffers as 1D)
    A_ = A.contiguous().view(-1)
    B_ = B.contiguous().view(-1)
    C_ = C.contiguous().view(-1)

    size = int(size)

    # Match CUDA grid configuration from the original code
    BLOCK = 1024     # blockDim.x
    NUM_BLOCKS = 256 # numBlocks in the original host launcher

    # Launch Triton kernel: one program per CUDA block (so grid = (256,))
    grid = (NUM_BLOCKS,)

    # Launch. BLOCK is passed as a constexpr parameter.
    _triton_kernel_impl[grid](A_, B_, C_, size, BLOCK=BLOCK)