import triton
import triton.language as tl
import torch

# Constants matching the CUDA launch configuration
BLOCK_SIZE = 1024
NUM_BLOCKS = 256
NUM_OUTER = 8
BLOCK_STRIDE = 262144  # 256 * BLOCK_SIZE

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)          # block index
    base = pid * BLOCK_SIZE              # start index for this block
    offset = tl.arange(0, BLOCK_SIZE)    # thread offsets within the block
    idx = base + offset                  # global thread indices

    for outer in range(NUM_OUTER):
        index = outer * BLOCK_STRIDE + idx
        mask = index < size
        a = tl.load(A_ptr + index, mask=mask)
        b = tl.load(B_ptr + index, mask=mask)
        tl.store(C_ptr + index, a + b, mask=mask)

def triton_kernel(A, B, C, size):
    """
    Wrapper that launches the Triton kernel with the same grid configuration
    as the original CUDA kernel.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, contiguous, on GPU).
    B : torch.Tensor
        Input tensor B (float32, contiguous, on GPU).
    C : torch.Tensor
        Output tensor C (float32, contiguous, on GPU).
    size : int
        Number of elements to process. The kernel will write to indices
        0 .. size-1, mirroring the bounds check in the CUDA version.
    """
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    grid = NUM_BLOCKS
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)