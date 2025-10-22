import torch
import triton
import triton.language as tl

# Triton kernel implementing the same computation as the original CUDA kernel.
# Each program (block) processes a logical chunk of 960 elements.
# Because tl.arange requires a power‑of‑two range, we generate a range of 1024
# and mask out the extra threads.
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK_SIZE: tl.constexpr):
    TILE_SIZE = 1024  # next power of two >= BLOCK_SIZE
    offs = tl.arange(0, TILE_SIZE)               # thread offsets within the program
    pid = tl.program_id(0)                       # block index
    # Global offsets for this program
    global_offs = pid * BLOCK_SIZE + offs
    # Mask: only the first BLOCK_SIZE threads are logical, and stay within `size`
    mask = (offs < BLOCK_SIZE) & (global_offs < size)

    a = tl.load(A_ptr + global_offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + global_offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + global_offs, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA kernel launch.
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
    # Input validation
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("All tensors must have dtype torch.float32.")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise RuntimeError("All tensors must be contiguous.")

    BLOCK_SIZE = 960  # matches __launch_bounds__(960) in the CUDA code

    # Compute number of programs (blocks) needed to cover `size`
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[(num_blocks,)](
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )