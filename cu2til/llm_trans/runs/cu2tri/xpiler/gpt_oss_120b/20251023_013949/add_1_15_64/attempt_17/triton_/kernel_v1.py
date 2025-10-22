import triton
import triton.language as tl
import torch

# Block size matches the CUDA launch bounds
BLOCK_SIZE = 960

@triton.jit
def _triton_kernel_impl(
    A_ptr: tl.pointer(tl.float32),
    B_ptr: tl.pointer(tl.float32),
    T_add_ptr: tl.pointer(tl.float32),
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that mirrors the original CUDA kernel:
    T_add[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
    It ignores blockIdx.x, exactly as the CUDA version does.
    """
    # Offsets within the block (0 .. BLOCK_SIZE-1)
    offs = tl.arange(0, BLOCK_SIZE)
    a = tl.load(A_ptr + offs)
    b = tl.load(B_ptr + offs)
    tl.store(T_add_ptr + offs, a + b)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point wrapper that reproduces the launch configuration of the
    original CUDA kernel.

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
        original CUDA kernel).
    """
    # Basic sanity checks – the original CUDA code expects raw device pointers.
    assert isinstance(A, torch.Tensor) and A.is_cuda, "A must be a CUDA tensor"
    assert isinstance(B, torch.Tensor) and B.is_cuda, "B must be a CUDA tensor"
    assert isinstance(C, torch.Tensor) and C.is_cuda, "C must be a CUDA tensor"

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