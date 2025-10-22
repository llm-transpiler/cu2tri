import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    A,          # float* __restrict__
    B,          # float* __restrict__
    T_add,      # float* __restrict__ (output)
    size,       # total number of elements to process
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)  # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(T_add + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton equivalent of the original CUDA kernel.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA)
    B : torch.Tensor
        Input tensor B (float32, CUDA)
    C : torch.Tensor
        Output tensor C (float32, CUDA)
    size : int
        Number of elements to process
    """
    # Input validation
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"

    size = int(size)  # ensure Python int

    BLOCK_SIZE = 1024  # matches __launch_bounds__(1024) in the CUDA kernel

    # Compute grid size (number of program instances)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32  # 1024 threads → 32 warps per block
    )