import torch
import triton
import triton.language as tl

# Block size must be a multiple of 32 and result in a power‑of‑two number of warps.
_BLOCK_SIZE = 1024  # 32 warps (1024 threads per block)

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                     # Block index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size                       # Guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size):
    # Ensure size is a Python int (handles torch scalar inputs)
    size = int(size)

    # Input validation (mirrors the original CUDA expectations)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), \
        "size exceeds tensor length"

    # Grid configuration
    grid = ((size + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)
    num_warps = _BLOCK_SIZE // 32  # Must be a power of two (32 here)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=num_warps,
    )