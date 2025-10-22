import torch
import triton
import triton.language as tl

# Number of threads per program (block)
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float32
    B_ptr,          # *float32
    C_ptr,          # *float32
    size,           # int64 scalar
    BLOCK_SIZE: tl.constexpr,
):
    # Program (block) ID
    pid = tl.program_id(0)
    # Offsets for this program
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask for out‑of‑bounds elements
    mask = offs < size
    # Load values (zero for masked‑off elements)
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    # Store the result
    tl.store(C_ptr + offs, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size int):
    """
    Triton wrapper that mimics the original CUDA kernel.
    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors residing on the same CUDA device.
    size : int
        Number of elements to process (must not exceed tensor length).
    """
    # Sanity checks
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise AssertionError("All tensors must be CUDA tensors")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise AssertionError("All tensors must be float32")
    if not (A.shape == B.shape == C.shape):
        raise AssertionError("All tensors must have the same shape")
    if size > A.numel():
        raise AssertionError("size exceeds tensor length")

    # Compute grid size to cover `size` elements
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)