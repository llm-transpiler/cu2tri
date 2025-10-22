import torch
import triton
import triton.language as tl

# Constants
_MAX_ELEMENTS = 4608  # Original CUDA kernel bound
_BLOCK_SIZE = 1024    # Threads per block (launch bounds)

@triton.jit
def _triton_kernel_impl(A_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton implementation of the sign kernel.
    Computes C[i] = sign(A[i]) for i < size (where size is clamped to
    the original kernel's maximum of 4608). Elements outside the mask are
    left unchanged.
    """
    pid = tl.program_id(0)  # Block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # Global indices

    # Mask for valid indices (size already clamped by the wrapper)
    mask = offsets < size

    # Load A with zero for masked-out lanes
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # Compute sign: 1.0 if a > 0, -1.0 if a < 0, else 0.0
    sign = tl.where(a > 0.0, 1.0, tl.where(a < 0.0, -1.0, 0.0))

    # Store result only for valid lanes
    tl.store(C_ptr + offsets, sign, mask=mask)


def triton_kernel(A: torch.Tensor, C: torch.Tensor, size: int) -> None:
    """
    Wrapper that launches the Triton kernel, mirroring the original CUDA API.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA) with at least `size` elements.
    C : torch.Tensor
        Output tensor (float32, CUDA) with at least `size` elements.
    size : int
        Number of elements to process (identical to the original CUDA kernel).
    """
    # Validation
    if not (A.is_cuda and C.is_cuda):
        raise RuntimeError("A and C must be CUDA tensors")
    if A.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("A and C must be torch.float32 tensors")
    if A.numel() < size or C.numel() < size:
        raise RuntimeError("Tensor sizes must be at least `size`")
    if size < 0:
        raise ValueError("size must be non‑negative")

    # Clamp the effective size to the original kernel's bound (4608)
    effective_size = min(size, _MAX_ELEMENTS)

    # Compute grid dimensions (same as original CUDA launch)
    grid = ((size + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A,                     # input pointer
        C,                     # output pointer
        effective_size,        # clamped size for the kernel
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=32,          # 1024 threads = 32 warps
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()