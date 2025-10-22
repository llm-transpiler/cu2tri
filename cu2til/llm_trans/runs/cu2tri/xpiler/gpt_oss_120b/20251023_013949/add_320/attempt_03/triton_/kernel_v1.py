import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK_SIZE: tl.constexpr):
    # Offsets for the threads inside a block (equivalent to threadIdx.x)
    offs = tl.arange(0, BLOCK_SIZE)
    # Load elements from A and B
    a = tl.load(A_ptr + offs)
    b = tl.load(B_ptr + offs)
    # Write the sum to T_add (C)
    tl.store(T_add_ptr + offs, a + b)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel with the same semantics as the original CUDA code.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32, residing on CUDA device.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32, residing on CUDA device.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32, residing on CUDA device.
    size : int
        Number of elements (must be >= 320 for correct behavior, matching the CUDA launch).
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Only torch.float32 tensors are supported"
    BLOCK_SIZE = 320  # launch bounds in the original CUDA kernel
    # Compute grid size exactly as the CUDA host code does
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)
    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE
    )