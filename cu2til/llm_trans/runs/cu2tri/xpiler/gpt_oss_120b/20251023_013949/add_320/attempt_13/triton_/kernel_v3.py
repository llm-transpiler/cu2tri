import torch
import triton
import triton.language as tl

# Use a power‑of‑two block size to satisfy tl.arange's requirement
BLOCK_SIZE = 256

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread offsets
    mask = offsets < size  # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton equivalent of the original CUDA kernel.
    Parameters:
        A (torch.Tensor): input float32 tensor on CUDA
        B (torch.Tensor): input float32 tensor on CUDA
        C (torch.Tensor): output float32 tensor on CUDA
        size (int): number of elements to process
    """
    # Basic validation (mirrors expectations of the CUDA version)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"

    # One‑dimensional grid of blocks
    grid = lambda meta: ((size + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    # Launch the Triton kernel with the compile‑time BLOCK_SIZE argument
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)