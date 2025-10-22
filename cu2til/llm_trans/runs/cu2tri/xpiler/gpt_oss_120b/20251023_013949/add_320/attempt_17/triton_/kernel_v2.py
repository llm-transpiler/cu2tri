import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, B, T_add, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE) + block_start
    mask = offs < size
    a = tl.load(A + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)
    tl.store(T_add + offs, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    # Validate inputs
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise ValueError("All tensors must be torch.float32")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise ValueError("All tensors must be contiguous")
    # Ensure size is an integer and within bounds
    if isinstance(size, torch.Tensor):
        size = size.item()
    size = int(size)
    if size < 0:
        raise ValueError("size must be non‑negative")
    if size > A.numel() or size > B.numel() or size > C.numel():
        raise ValueError("size exceeds tensor length")
    if size == 0:
        return
    # Triton kernel configuration
    BLOCK_SIZE = 256  # power‑of‑two for tl.arange
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)
    # Launch kernel
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Synchronize to make kernel effects visible before returning
    torch.cuda.synchronize(A.device)