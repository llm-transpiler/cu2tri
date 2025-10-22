import torch
import triton
import triton.language as tl

# Compile‑time block size (matches the CUDA launch bounds)
BLOCK_SIZE = 320

@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float32
    B_ptr,          # *float32
    C_ptr,          # *float32
    size,           # i32
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that performs element‑wise addition:
        C[i] = A[i] + B[i]   for i in [0, size)
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # linear thread indices
    mask = offsets < size                       # guard against OOB

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int) -> None:
    """
    Wrapper that mimics the original CUDA launch configuration.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32) on CUDA device.
    B : torch.Tensor
        Input tensor (float32) on CUDA device.
    C : torch.Tensor
        Output tensor (float32) on CUDA device.
    size : int
        Number of elements to process.
    """
    # Basic validation
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must have dtype torch.float32.")
    if size < 0:
        raise ValueError("size must be non‑negative.")

    # Ensure contiguous memory layout
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # Compute grid dimensions (same as CUDA launch)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Synchronize to guarantee completion before returning
    torch.cuda.synchronize()