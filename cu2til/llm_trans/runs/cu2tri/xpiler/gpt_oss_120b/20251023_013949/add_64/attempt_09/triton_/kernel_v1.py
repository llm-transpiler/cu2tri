import torch
import triton
import triton.language as tl

# Block size matching the CUDA launch bounds
BLOCK_SIZE = 64

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that adds two vectors elementwise.
    Parameters
    ----------
    A_ptr, B_ptr, C_ptr : pointers to float32 tensors
    size : total number of elements to process
    BLOCK_SIZE : compile‑time constant (64)
    """
    pid = tl.program_id(0)                     # block index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int) -> None:
    """
    Entry‑point that mimics the original CUDA kernel signature.
    Parameters
    ----------
    A, B, C : torch tensors of dtype torch.float32 on the same CUDA device
    size    : number of elements to process (must be <= len(A), len(B), len(C))
    """
    # Basic validation
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), \
        "size must not exceed tensor lengths"

    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        # Optional performance hints
        num_warps=4,
        num_stages=2,
    )