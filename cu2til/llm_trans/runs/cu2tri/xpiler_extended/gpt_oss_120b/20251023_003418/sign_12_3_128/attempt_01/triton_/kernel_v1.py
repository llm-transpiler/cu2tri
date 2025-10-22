import torch
import triton
import triton.language as tl

# Original CUDA kernel processes at most 4608 elements.
_MAX_ELEMENTS = 4608
_BLOCK_SIZE = 1024  # matches the CUDA launch bounds


@triton.jit
def _triton_kernel_impl(A_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton implementation of the sign kernel.
    For each i < min(size 4608):
        C[i] =  1.0 if A[i] > 0
               -1.0 if A[i] < 0
                0.0 otherwise
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices

    # Apply both the user‑provided size bound and the original 4608 bound.
    mask = (offsets < size) & (offsets < _MAX_ELEMENTS)

    # Load A (masked)
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # Compute sign
    sign = tl.where(a > 0.0, 1.0,
                    tl.where(a < 0.0, -1.0, 0.0))

    # Store result (masked)
    tl.store(C_ptr + offsets, sign,=maskdef triton_kernel(A: torch.Tensor, C: torch.Tensor, size: int) -> None:
    """
    Wrapper that launches the Triton kernel.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA) with at least `size` elements.
    C : torch.Tensor
        Output tensor (float32, CUDA) with at least `size` elements.
    size : int
        Number of elements to process (mirrors the original CUDA API).
    """
    # Basic validation
    if not (A.is_cuda and C.is_cuda):
        raise RuntimeError("A and C must be CUDA tensors")
    if A.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("A and C must be torch.float32 tensors")
    if A.numel() < size or C.numel() < size:
        raise RuntimeError("Tensor sizes must be at least `size`")
    if size < 0:
        raise ValueError("size must be non‑negative")

    # Compute grid dimensions (same strategy as the CUDA launch)
    grid = ((size + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A,                     # pointer to input
        C,                     # pointer to output
        size,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=32,          # 1024 threads = 32 warps
    )
    # Synchronize to ensure kernel completion before returning
    torch.cuda.synchronize()