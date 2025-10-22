import torch
import triton
import triton.language as tl

# Triton kernel implementation
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float32
    B_ptr,          # *float32
    T_add_ptr,      # *float32
    size,           # i32
    BLOCK_SIZE: tl.constexpr
):
    """
    Element‑wise addition: T_add[i] = A[i] + B[i] for i < size.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices
    mask = offsets < size

    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    tl.store(T_add_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel.
    Parameters
    ----------
    A, B, C : torch.Tensor
        Must be contiguous, CUDA, dtype=torch.float32.
    size : int
        Number of elements to process.
    """
    # Validate inputs
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be on CUDA device")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("All tensors must be torch.float32")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()

    BLOCK_SIZE = 64
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4  # typical for 64‑element blocks
    )