import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK_SIZE: tl.constexpr):
    # Offsets of threads a block (0 .. BLOCK_SIZE-1)
    offs = tl.arange(0, BLOCK_SIZE)
    # Global index for each thread
    idx = tl.program_id(0) * BLOCK_SIZE + offs
    # Guard against out‑of‑bounds accesses
    mask = idx < size
    # Load inputs (masked)
    a = tl.load(A_ptr + idx, mask=mask)
    b = tl.load(B_ptr + idx, mask=mask)
    # Store the sum (masked)
    tl.store(T_add_ptr + idx, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton implementation matching the original CUDA kernel.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32 on CUDA.
    size : int
        Number of elements to process.
    """
    # Ensure `size` is a Python int (handles torch scalar inputs)
    size = int(size)

    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Only torch.float32 tensors are supported"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "Tensors must be contiguous"
    assert size == A.numel() == B.numel() == C.numel(), \
        "size must match the number of elements in the tensors"

    # Triton block size (must be a power of two for tl.arange)
    BLOCK_SIZE = 256

    # Compute grid dimensions
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    if num_blocks == 0:
        return
    grid = (num_blocks,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()