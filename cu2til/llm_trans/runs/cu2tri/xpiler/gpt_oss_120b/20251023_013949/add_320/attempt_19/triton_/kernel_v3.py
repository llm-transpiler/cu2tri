import torch
import triton
import triton.language as tl

# Triton kernel that performs element‑wise addition.
# It mimics the CUDA kernel which uses 320 threads per block.
# Since tl.arange requires a power‑of‑two range, we process the
# 320 elements in chunks of 64 (a power of two) inside a static loop.
@triton.jit
def _triton_kernel_impl(
    A,          # *float32
    B,          # *float32
    T_add,      # *float32
    BLOCK_SIZE: tl.constexpr,  # compile‑time constant (320)
):
    # Base offset for this block (program)
    block_start = tl.program_id(0) * BLOCK_SIZE

    # Process BLOCK_SIZE elements in 64‑element chunks.
    # BLOCK_SIZE is a constexpr, so the loop is unrolled at compile time.
    for i in range(0, BLOCK_SIZE, 64):  # 0, 64, 128, 192, 256
        offs = block_start + i + tl.arange(0, 64)  # offsets for this chunk
        a = tl.load(A + offs)
        b = tl.load(B + offs)
        tl.store(T_add + offs, a + b)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point matching the original CUDA signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D contiguous float32 tensors on the GPU.
    size : int
        Logical vector length (kept for API compatibility; not used inside the kernel).
    """
    # Basic validation
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"

    BLOCK_SIZE = 320

    # Compute grid size exactly as in the CUDA launch.
    grid_x = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_x,)

    # Launch the Triton kernel.
    # num_warps must be a power of two; 2 warps (64 threads) are sufficient
    # because the kernel internally iterates over 5 chunks of 64 threads.
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=2,
    )