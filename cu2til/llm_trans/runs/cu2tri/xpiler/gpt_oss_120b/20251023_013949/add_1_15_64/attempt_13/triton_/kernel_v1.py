import torch
import triton
import triton.language as tl

# Block size matches the CUDA launch bounds
BLOCK_SIZE = 960

@triton.jit
def _triton_kernel_impl(A, B, T_add, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that performs element‑wise addition of two vectors.
    Mirrors the behaviour of the original CUDA kernel `_cuda_kernel_impl`.
    """
    # Thread index within the block (0 … BLOCK_SIZE‑1)
    offs = tl.arange(0, BLOCK_SIZE)

    # Load a single element from each input
    a = tl.load(A + offs)
    b = tl.load(B + offs)

    # Store the sum
    tl.store(T_add + offs, a + b)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original `cuda_kernel` signature.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) on CUDA device, dtype torch.float32.
    B : torch.Tensor
        Input tensor of shape (size,) on CUDA device, dtype torch.float32.
    C : torch.Tensor
        Output tensor of shape (size,) on CUDA device, dtype torch.float32.
    size : int
        Number of elements (used only for grid calculation, identical to CUDA).
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Only float32 tensors are supported"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor sizes must be at least `size`"

    # Compute grid dimensions – same formula as the CUDA launch
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel.
    # `num_warps` = BLOCK_SIZE // 32 ensures 960 threads per block.
    _triton_kernel_impl[grid](
        A, B, C,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32,
        num_stages=2  # typical for simple kernels
    )