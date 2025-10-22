import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes C = A + B elementwise for `size` elements.
    Mirrors the behavior of the original CUDA kernel.
    """
    pid = tl.program_id(0)  # block index
    # Compute absolute offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask out-of-range indices
    mask = offsets < size

    # Load A and B with mask (out-of-bounds loads return 0.0)
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    # Store the result
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that mimics the original CUDA launch configuration.
    Parameters:
        A (torch.Tensor): Input tensor of shape (size,) on CUDA, dtype float32.
        B (torch.Tensor): Input tensor of shape (size,) on CUDA, dtype float32.
        C (torch.Tensor): Output tensor of shape (size,) on CUDA, dtype float32.
        size (int): Number of elements to process.
    """
    # Sanity checks (optional but helpful)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor sizes must be at least `size`"

    BLOCK_SIZE = 1024  # matches __launch_bounds__(1024) in the CUDA code
    # Compute number of blocks needed to cover `size` elements
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[(num_blocks,)](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

    # Optional: synchronize for debugging / timing
    # torch.cuda.synchronize()