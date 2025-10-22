import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise addition with a hardcoded bound of 2304.
@triton.jit
def _triton_kernel_impl(A, B, T_add, BLOCK_SIZE: tl.constexpr):
    # program_id(0) corresponds to the block index in the 1‑D grid.
    pid = tl.program_id(0)
    # Compute the absolute indices for this block.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Hard‑coded guard identical to the original CUDA kernel.
    mask = offsets < 2304
    # Load A and B with masking; out‑of‑range loads return 0.0 (won’t affect the result).
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    # Store the sum into T_add respecting the same mask.
    tl.store(T_add + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA launch configuration.
    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float tensors residing on the same CUDA device.
    size : int
        Logical size of the vectors (used only to compute grid dimensions,
        mirroring the CUDA host code). The kernel itself uses a hard‑coded
        bound of 2304, matching the original implementation.
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32
    # Ensure tensors are contiguous for pointer arithmetic.
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    BLOCK_SIZE = 1024
    # Compute number of blocks exactly as in the CUDA host code.
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)
    # Launch the Triton kernel.
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE=BLOCK_SIZE)