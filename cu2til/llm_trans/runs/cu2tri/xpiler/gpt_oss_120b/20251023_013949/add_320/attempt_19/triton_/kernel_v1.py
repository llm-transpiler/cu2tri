import torch
import triton
import triton.language as tl

# Triton kernel implementing the same elementwise addition as the CUDA kernel.
# The kernel writes to T_add[threadIdx.x] for each block, ignoring blockIdx.x,
# which matches the original CUDA kernel's semantics.
@triton.jit
def _triton_kernel_impl(
    A,          # *float32
    B,          # *float32
    T_add,      # *float32
    BLOCK_SIZE: tl.constexpr,  # compile‑time constant: 320
):
    # Offsets of the threads within the block (0 … BLOCK_SIZE‑1)
    offs = tl.arange(0, BLOCK_SIZE)

    # Load values from A and B. No bounds checking – mirrors the CUDA code.
    a = tl.load(A + offs)
    b = tl.load(B + offs)

    # Store the sum into T_add.
    tl.store(T_add + offs, a + b)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original `cuda_kernel` signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D contiguous float32 tensors residing on the GPU.
    size : int
        Logical size of the vectors (unused inside the kernel, kept for API compatibility).
    """
    # Basic sanity checks – they are cheap and help catch misuse early.
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"

    BLOCK_SIZE = 320

    # Compute grid exactly as in the CUDA launch.
    # Each program (block) processes BLOCK_SIZE elements, but the kernel
    # ignores the block index when indexing, reproducing the original behavior.
    grid_x = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_x,)

    # Launch the Triton kernel.
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        BLOCK_SIZE=BLOCK_SIZE,
        # optional: set num_warps to match the launch bounds of 320 threads.
        # 10 warps * 32 = 320 threads.
        num_warps=10,
    )