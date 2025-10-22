# Triton implementation of elementwise addition kernel
import torch
import triton
import triton.language as tl

# Block size matches the CUDA launch bounds (1024 threads per block)
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float32
    B_ptr,          # *float32
    C_ptr,          # *float32
    size,           # i32 scalar
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute C = A + B element‑wise for `size` elements.
    Mirrors the behaviour of the original CUDA kernel.
    """
    pid = tl.program_id(0)                # block index
    block_offset = pid * BLOCK_SIZE
    offsets = block_offset + tl.arange(0, BLOCK_SIZE)

    # Guard against out‑of‑bounds accesses
    mask = offsets < size

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int) -> None:
    """
    Entry‑point that mimics the original `cuda_kernel` signature.
    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D tensors of type torch.float32 residing on the CUDA device.
    size : int
        Number of elements to process (must be <= len(A), len(B), len(C)).
    """
    # Basic sanity checks – keep them lightweight for performance
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "Tensors must be contiguous"
    # Compute grid dimension (one‑dimensional launch)
    grid = lambda meta: ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE
    )