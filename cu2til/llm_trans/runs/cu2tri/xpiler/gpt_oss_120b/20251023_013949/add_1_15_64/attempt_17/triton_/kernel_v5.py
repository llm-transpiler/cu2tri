import torch
import triton
import triton.language as tl

# Use a power‑of‑two block size to satisfy tl.arange constraints.
BLOCK_SIZE = 256

@triton.jit
def _triton_kernel_impl(
    A_ptr,          # pointer to input A
    B_ptr,          # pointer to input B
    T_add_ptr,      # pointer to output (C)
    N: tl.int32,    # total number of elements to process
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that launches the Triton kernel with the same semantics as the
    original CUDA kernel.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError("All tensors must have dtype torch.float32")
    if A.shape != B.shape or A.shape != C.shape:
        raise ValueError("All tensors must have the same shape")
    if size < 0:
        raise ValueError("size must be non‑negative")
    if size > A.numel():
        raise ValueError("size exceeds number of elements in the tensors")
    if size == 0:
        return

    # ------------------------------------------------------------------
    # Grid configuration (mirroring the original CUDA launch)
    # ------------------------------------------------------------------
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    # 256 threads per block = 8 warps (power of two, required by Triton)
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )