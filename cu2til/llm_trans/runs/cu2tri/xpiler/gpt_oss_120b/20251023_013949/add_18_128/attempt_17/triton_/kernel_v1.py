import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Only compute for indices < 2304 (mirrors original CUDA condition)
    mask = offsets < 2304
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(T_add_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton equivalent of the original CUDA kernel launcher.
    Parameters
    ----------
    A, B, C : torch.Tensor
        Input and output tensors (float32) residing on the CUDA device.
    size : int
        Number of elements to process (used only for grid sizing, the kernel
        itself caps work to 2304 elements as in the CUDA version).
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    # Ensure tensors are contiguous for safe pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    # Compute launch configuration (mirrors CUDA launch)
    BLOCK_SIZE = 1024
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)
    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE=BLOCK_SIZE)