import torch
import triton
import triton.language as tl

# Triton kernel implementing the same elementwise addition as the CUDA kernel.
# It writes to T_add[threadIdx.x] for each block, matching the original CUDA semantics.
@triton.jit
def _triton_kernel_impl(
    A,          # *float32
    B,          # *float32
    T_add,      # *float32
    BLOCK_SIZE: tl.constexpr,
):
    # Offsets of the threads within the block (0 … BLOCK_SIZE‑1)
    offs = tl.arange(0, BLOCK_SIZE)

    # Load values from A and B (no bounds checking – mirrors the CUDA code)
    a = tl.load(A + offs)
    b = tl.load(B + offs)

    # Store the sum into T_add
    tl.store(T_add + offs, a + b)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original `cuda_kernel` signature:
        void cuda_kernel(float *A, float *B, float *C, int size)
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device"
    assert (
        A.dtype == torch.float32
        and B.dtype == torch.float32
        and C.dtype == torch.float32
    ), "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "All tensors must be contiguous"

    BLOCK_SIZE = 320

    # Compute grid exactly as in the CUDA launch.
    grid_x = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_x,)

    # Launch the Triton kernel.
    # num_warps must be a power of 2; we choose 16 (>= ceil(320/32)=10).
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )