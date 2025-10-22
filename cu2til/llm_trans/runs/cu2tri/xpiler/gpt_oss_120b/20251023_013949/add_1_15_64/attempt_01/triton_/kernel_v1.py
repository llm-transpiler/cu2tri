import torch
import triton
import triton.language as tl

# Compile-time block size (must be a multiple of 32 for Hopper)
BLOCK_SIZE = 960

@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float32
    B_ptr,          # *float32
    C_ptr,          # *float32
    N: tl.int32,    # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that computes C[i] = A[i] + B[i] for i in [0, N).
    Mirrors the behaviour of the original CUDA kernel that used a
    thread block of size 960.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original ``cuda_kernel`` signature.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) on CUDA device, dtype torch.float32.
    B : torch.Tensor
        Input tensor of shape (size,) on CUDA device, dtype torch.float32.
    C : torch.Tensor
        Output tensor of shape (size,) on CUDA device, dtype torch.float32.
    size : int
        Number of elements to process.
    """
    # Basic validation (mirrors the expectations of the CUDA version)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor storage must be at least `size` elements"

    # Compute grid dimensions (equivalent to the CUDA launch configuration)
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=30,  # 30 warps * 32 = 960 threads per block
    )