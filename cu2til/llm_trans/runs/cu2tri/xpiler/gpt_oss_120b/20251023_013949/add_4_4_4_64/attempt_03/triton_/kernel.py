import torch
import triton
import triton.language as tl

# Triton kernel that reproduces the original CUDA implementation.
@triton.jit
def _triton_kernel_impl(A, B, C, BLOCK_SIZE: tl.constexpr):
    """
    Element‑wise addition of two float32 vectors.
    The kernel mirrors the CUDA version, including the hard‑coded 4096‑element bound.
    """
    pid = tl.program_id(0)                                 # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global element indices
    mask = offsets < 4096                                   # original bound

    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point matching the original `cuda_kernel` signature.
    Parameters
    ----------
    A, B, C : torch.Tensor
        Input and output tensors (float32) residing on the same CUDA device.
    size : int
        Logical size used to compute the launch grid (identical to the CUDA wrapper).
    """
    # Validate device and dtype.
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be on a CUDA device.")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("All tensors must have dtype torch.float32.")

    # Flatten to 1‑D contiguous tensors to match the pointer semantics of the CUDA kernel.
    A_flat = A.contiguous().view(-1)
    B_flat = B.contiguous().view(-1)
    C_flat = C.contiguous().view(-1)

    BLOCK_SIZE = 1024
    # Compute grid size exactly as the original CUDA launch configuration.
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel. `num_warps=8` provides ample parallelism on Hopper GPUs.
    _triton_kernel_impl[grid](A_flat, B_flat, C_flat, BLOCK_SIZE, num_warps=8)