import torch
import triton
import triton.language as tl

# Compile‑time block size (matches the original CUDA launch bounds)
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(
    A_ptr,                # *float32
    compute_ptr,          # *float32
    size,                 # i32 scalar
    BLOCK_SIZE: tl.constexpr  # compile‑time constant
):
    """
    Triton implementation of the original CUDA kernel.

    For each global index i = block_id * BLOCK_SIZE + lane_id,
    if i < min(size, 4608) compute:
        compute[i] = 1.0 / (1.0 + exp(-A[i]))
    """
    pid = tl.program_id(axis=0)               # block index
    offsets = tl.arange(0, BLOCK_SIZE)        # thread indices within the block
    idx = pid * BLOCK_SIZE + offsets           # global index

    # Preserve the original hard‑coded bound (4608) while also respecting the
    # user‑provided ``size`` to avoid out‑of‑bounds accesses.
    mask = (idx < size) & (idx < 4608)

    a = tl.load(A_ptr + idx, mask=mask, other=0.0)
    # sigmoid: 1 / (1 + exp(-x))
    result = 1.0 / (1.0 + tl.exp(-a))
    tl.store(compute_ptr + idx, result, mask=mask)


def triton_kernel(A: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point mirroring the original ``cuda_kernel`` signature.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32) on CUDA device.
    C : torch.Tensor
        Output tensor (float32) on CUDA device.
    size : int
        Logical length of the vectors (used for grid calculation).
    """
    # Sanity checks
    if not (A.is_cuda and C.is_cuda):
        raise RuntimeError("Both A and C must be CUDA tensors.")
    if A.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("Tensors must be of type torch.float32.")
    if A.numel() < size or C.numel() < size:
        raise RuntimeError("Tensor storage must be at least ``size`` elements.")

    # Grid configuration (same as the original CUDA launch)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
    )