import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel (named exactly as required)
# ----------------------------------------------------------------------
BLOCK_SIZE = 320  # matches the CUDA launch bounds

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Element‑wise addition: C = A + B
    Each program (block) processes BLOCK_SIZE contiguous elements.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread offsets
    mask = offsets < size                       # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

# ----------------------------------------------------------------------
# Python wrapper (entry point) – identical signature to the CUDA version
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton equivalent of the original `cuda_kernel`.
    Parameters:
        A (torch.Tensor): input tensor of shape (>=size,) on CUDA, dtype float32
        B (torch.Tensor): input tensor of shape (>=size,) on CUDA, dtype float32
        C (torch.Tensor): output tensor of shape (>=size,) on CUDA, dtype float32
        size (int): number of elements to process
    """
    # Basic sanity checks (mirrors expectations of the CUDA code)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    # Compute grid dimensions (number of blocks)
    grid = lambda meta: ((size + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK)