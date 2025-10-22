import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Triton kernel: elementwise addition C = A + B
# -------------------------------------------------------------------------
@triton.jit(num_warps=8)
def _triton_kernel_impl(
    A_ptr,                     # *float32
    B_ptr,                     # *float32
    C_ptr,                     # *float32
    size,                      # i32
    BLOCK_SIZE: tl.constexpr   # compile‑time constant
):
    pid = tl.program_id(0)                     # 1‑D grid
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


# -------------------------------------------------------------------------
# Wrapper matching the original CUDA kernel signature
# -------------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton implementation of the original CUDA elementwise add kernel.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of dtype torch.float32 on CUDA.
    size : int
        Number of elements to process (must be ≤ len(A), len(B), len(C)).
    """
    # Basic sanity checks (mirrors the __restrict__ contract)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.numel() >= size and B.numel() >= andel() >= size, \
        "Tensor sizes must be least 'size'"

    BLOCK_SIZE = 1024  # matches the original blockDim.x
    grid = (triton.cdiv(size, BLOCK_SIZE),)  # 1‑D grid

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE
    )