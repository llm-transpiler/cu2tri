import torch
import triton
import triton.language as tl

# Compile-time block size (must be a multiple of 32)
_BLOCK_SIZE = 960

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes C[i] = A[i] + B[i] for i in [0, size).
    """
    pid = tl.program_id(0)                      # Program (block) ID
    block_start = pid * BLOCK_SIZE               # Starting index for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Indices handled by this block
    mask = offsets < size                         # Guard against out-of-bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton implementation of the original CUDA kernel:
    C[i] = A[i] + B[i] for i in [0, size).

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
    # Validate inputs
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), \
        "size exceeds tensor length"

    # Compute grid dimensions
    grid = ((size + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=_BLOCK_SIZE // 32,  # 960/32 = 30 warps
    )