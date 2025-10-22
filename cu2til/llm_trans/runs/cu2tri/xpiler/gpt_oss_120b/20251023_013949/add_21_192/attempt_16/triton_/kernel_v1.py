import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, B, T_add, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < 4032
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(T_add + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton equivalent of the original CUDA kernel.
    Performs element‑wise addition of A and B into C for indices < 4032.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA) of length at least `size`.
    B : torch.Tensor
        Input tensor (float32, CUDA) of length at least `size`.
    C : torch.Tensor
        Output tensor (float32, CUDA) of length at least `size`.
    size : int
        Logical size used only for grid calculation (mirrors the CUDA launch).
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "All tensors must be float32"
    BLOCK_SIZE = 1024
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)
    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)