import torch
import triton
import triton.language as tl

# Block size matches the CUDA launch bounds
BLOCK_SIZE = 960

@triton.jit
def _triton_kernel_impl(
    A_ptr,
    B_ptr,
    T_add_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that reproduces the original CUDA kernel:
    T_add[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
    The kernel deliberately ignores blockIdx.x, exactly like the CUDA version.
    """
    # Offsets for the 960 threads in the block
    offs = tl.arange(0, BLOCK_SIZE)
    a = tl.load(A_ptr + offs)
    b = tl.load(B_ptr + offs)
    tl.store(T_add_ptr + offs, a + b)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel with the same launch configuration
    as the original CUDA code.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA).
    B : torch.Tensor
        Input tensor B (float32, CUDA).
    C : torch.Tensor
        Output tensor where the sum is written (float32, CUDA).
    size : int
        Number of elements to process (mirrors the `size` argument of the
        original CUDA kernel).  The kernel itself does not use this value for
        indexing, matching the original behavior.
    """
    # Basic sanity checks – the original CUDA code expects raw device pointers.
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError("All tensors must be of type torch.float32")

    # Compute grid dimensions identical to the CUDA launch:
    #   dim3 blockSize(960);
    #   dim3 numBlocks((size + 960 - 1) / 960);
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel.
    # 960 threads = 30 warps (30 * 32 = 960)
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=30,
    )