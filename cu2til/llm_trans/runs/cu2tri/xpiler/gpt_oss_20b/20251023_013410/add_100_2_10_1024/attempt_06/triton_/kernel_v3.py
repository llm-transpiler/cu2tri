import tr
import triton.language as tl
import torch

# Kernel implementation
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    # block index and thread index
    block_idx = tl.program_id(axis=0)
    thread_idx = tl.arange(0, BLOCK_SIZE)

    # base index for this block/thread
    base = block_idx * BLOCK_SIZE + thread_idx

    # constant stride for the outer loop
    STRIDE = 262144  # 256 * 1024

    # outer loop (8 iterations)
    for outer in range(8):
        idx = outer * STRIDE + base
        mask = idx < size
        a = tl.load(A_ptr + idx, mask=mask)
        b = tl.load(B_ptr + idx, mask=mask)
        tl.store(C_ptr + idx, a + b, mask=mask)

# Wrapper function
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Launches the Triton kernel with the same grid configuration as the original CUDA kernel.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, contiguous, on GPU).
    B : torch.Tensor
        Input tensor B (float32, contiguous, on GPU).
    C : torch.Tensor
        Output tensor C (float32, contiguous, on GPU).
    size : int
        Number of elements to process (should be <= 2048000).
    """
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # Original CUDA launch configuration: 256 blocks, 1024 threads per block
    grid = 256
    BLOCK_SIZE = 1024

    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)