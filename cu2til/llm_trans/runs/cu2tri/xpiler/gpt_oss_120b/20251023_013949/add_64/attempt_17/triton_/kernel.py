import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                      # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread offsets within block
    mask = offsets < N                           # guard against out‑of‑bounds
    a = tl.load(A_ptr + offsets, mask=mask)      # load A
    b = tl.load(B_ptr + offsets, mask=mask)      # load B
    tl.store(C_ptr + offsets, a + b, mask=mask)  # store A+B into C

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """Launches the Triton kernel that computes C = A + B element‑wise.

    Args:
        A (torch.Tensor): Input tensor of shape (size,) and dtype torch.float32.
        B (torch.Tensor): Input tensor of shape (size,) and dtype torch.float32.
        C (torch.Tensor): Output tensor of shape (size,) and dtype torch.float32.
        size (int): Number of elements to process.
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    BLOCK_SIZE = 64
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the Triton kernel
    _triton_kernel_impl[(num_blocks,)](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)