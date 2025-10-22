import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise addition: C[i] = A[i] + B[i] for i in [0, N)
    """
    pid = tl.program_id(0)                         # block index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # offsets for this block
    mask = offs < N                                 # guard for out‑of‑bounds

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(C_ptr + offs, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA kernel launch.
    Computes C[i] = A[i] + B[i] for i in [0, size).
    """
    # Validation
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be torch.float32.")
    if size < 0:
        raise ValueError("size must be non‑negative.")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError("Tensor length is smaller than the specified size.")

    BLOCK_SIZE = 256                     # power‑of‑two block size (multiple of 32)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)  # 1‑D grid

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,                     # 8 warps = 256 threads, power of two as required
    )