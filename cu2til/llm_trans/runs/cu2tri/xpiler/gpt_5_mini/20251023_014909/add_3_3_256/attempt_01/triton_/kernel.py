import math
import torch
import triton
import triton.language as tl

# Triton kernel: name must be exactly _triton_kernel_impl
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, BLOCK: tl.constexpr):
    """
    Triton implementation of the CUDA kernel. Mirrors the CUDA index math and
    the hard-coded bound-check against 2304 present in the original CUDA kernel.
    BLOCK is a compile-time constant (1024).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    # Replicate the original CUDA conditional: index < 2304
    mask = offsets < 2304
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(T_add_ptr + offsets, c, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Entry-point wrapper that matches the original CUDA external function signature:
      cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    - A, B, C: torch.cuda.FloatTensor (device tensors). Must be on CUDA.
    - size: int (used to configure grid size exactly like the original wrapper)

    Behavior is identical to the original CUDA wrapper + kernel:
    - grid is computed as ceil(size / 1024)
    - the kernel itself writes only for indices < 2304 (exactly as in the CUDA kernel)
    """
    # Basic checks to ensure correct types (keep behavior predictable)
    if not (torch.is_tensor(A) and torch.is_tensor(B) and torch.is_tensor(C)):
        raise TypeError("A, B, C must be torch tensors")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must be torch.float32 tensors")
    if int(size) < 0:
        raise ValueError("size must be non-negative")

    BLOCK = 1024  # match the CUDA block size
    num_blocks = (int(size) + BLOCK - 1) // BLOCK
    if num_blocks <= 0:
        return  # nothing to launch

    # Launch Triton kernel. This mirrors: _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, B, C);
    _triton_kernel_impl[(num_blocks,)](A, B, C, BLOCK=BLOCK)