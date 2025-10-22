import torch
import triton
import triton.language as tl

# Compile‑time block size (must match the CUDA launch configuration)
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(A, B, T_add, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that mirrors the original CUDA kernel.
    It adds A and B element‑wise into T_add, but only for indices
    < 4096 (hard‑coded bound from the CUDA implementation).
    """
    pid = tl.program_id(0)                     # block index
    offsets = tl.arange(0, BLOCK_SIZE)         # thread indices within the block
    idx = pid * BLOCK_SIZE + offsets            # global linear index

    # Hard‑coded bound: 4 * 1024 = 4096 (identical to the CUDA condition)
    mask = idx < (4 * BLOCK_SIZE)

    a = tl.load(A + idx, mask=mask, other=0.0)
    b = tl.load(B + idx, mask=mask, other=0.0)
    tl.store(T_add + idx, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that launches the Triton kernel with the same semantics
    as the original CUDA kernel.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32) of length at least `size`.
    B : torch.Tensor
        Input tensor (float32) of length at least `size`.
    C : torch.Tensor
        Output tensor (float32) where the result is stored.
    size : int
        Number of elements to process (used only for grid sizing,
        mirroring the CUDA launch configuration).
    """
    # Sanity checks – the original CUDA kernel expects contiguous float32 buffers.
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous()
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32

    # Grid configuration identical to the CUDA launch (ceil division).
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel.
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE=BLOCK_SIZE)