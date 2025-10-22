import torch
import triton
import triton.language as tl

# Use a block size that yields a power‑of‑two number of warps (required by Triton).
BLOCK_SIZE = 1024  # 32 warps × 32 threads = 1024 threads per block


@triton.jit
def _triton_kernel_impl(A, B, T_add, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel performing element‑wise addition of two vectors.
    Mirrors the behaviour of the original CUDA kernel.
    """
    pid = tl.program_id(0)                # Block index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard against out‑of‑bounds accesses.
    mask = offsets < size

    a = tl.load(A + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)

    tl.store(T_add + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original `cuda_kernel` signature.
    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors residing on the CUDA device.
    size : int
        Number of elements to process.
    """
    # Sanity checks.
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Only float32 tensors are supported"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor sizes must be at least `size`"

    if size == 0:
        return  # Nothing to do.

    # Compute grid dimensions – same formula as the CUDA launch.
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel.
    # num_warps must be a power of two; BLOCK_SIZE // 32 = 32 satisfies this.
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32,
        num_stages=2,
    )