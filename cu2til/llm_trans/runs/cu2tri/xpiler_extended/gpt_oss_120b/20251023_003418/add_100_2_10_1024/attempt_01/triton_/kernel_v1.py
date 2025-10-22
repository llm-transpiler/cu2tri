import torch
import triton
import triton.language as tl

# Number of threads per program (block)
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(
    A_ptr,               # *float32
    B_ptr,               # *float32
    C_ptr,               # *float32
    size,                # scalar (int64)
    BLOCK_SIZE: tl.constexpr
):
    # Program (block) ID
    pid = tl.program_id(0)
    # Offsets for this program
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE, dtype=tl.int64)
    # Mask for out‑of‑bounds elements
    mask = offs
    # Load (zero for masked‑off elements)
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    # Store the result
    tl.store(C_ptr + offs, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton wrapper that mimics the original CUDA kernel.
    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors residing on the same CUDA device.
    size : int
        Number of elements to process (equivalent to the bound in the CUDA kernel).
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Tensors must be float32"
    assert A.shape == B.shape == C.shape, "All tensors must have the same shape"
    # Ensure size does not exceed tensor length
    assert size <= A.numel(), "size exceeds tensor length"

    # Compute grid size to cover `size` elements
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    # Launch kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)