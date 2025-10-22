import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing the same element‑wise addition as the CUDA kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,                     # tl.pointer(tl.float32)
    B_ptr,                     # tl.pointer(tl.float32)
    C_ptr,                     # tl.pointer(tl.float32)
    size: tl.int32,            # total number of elements to process
    BLOCK_SIZE: tl.constexpr,  # number of elements each block is responsible for (320)
    THREADS_PER_BLOCK: tl.constexpr,  # power‑of‑two thread count for tl.arange (256)
):
    pid = tl.program_id(0)                     # block index
    base = pid * BLOCK_SIZE                    # first element index for this block

    # Number of iterations needed to cover BLOCK_SIZE elements with THREADS_PER_BLOCK threads
    num_iters = (BLOCK_SIZE + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    for i in range(num_iters):
        # Global offsets for this chunk
        offset = base + i * THREADS_PER_BLOCK + tl.arange(0, THREADS_PER_BLOCK)

        # Mask to stay within the block's range and overall size
        mask = (offset < size) & (offset < base + BLOCK_SIZE)

        a = tl.load(A_ptr + offset, mask=mask, other=0.0)
        b = tl.load(B_ptr + offset, mask=mask, other=0.0)
        tl.store(C_ptr + offset, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton entry point mirroring the original CUDA kernel.

    Parameters
    ----------
    A, B, C : torch.Tensor
        Input/output tensors (float32) residing on the CUDA device.
    size : int
        Number of elements to process.
    """
    # ------------------------------------------------------------------
    # Input validation (mirrors typical CUDA expectations)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert (
        A.dtype == torch.float32
        and B.dtype == torch.float32
        and C.dtype == torch.float32
    ), "All tensors must be torch.float32"

    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # ------------------------------------------------------------------
    # Grid configuration – identical to the CUDA launch parameters
    # ------------------------------------------------------------------
    BLOCK_SIZE = 320               # matches __launch_bounds__(320) in the CUDA kernel
    THREADS_PER_BLOCK = 256        # power‑of‑two thread count required by tl.arange
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        THREADS_PER_BLOCK=THREADS_PER_BLOCK,
        num_warps=THREADS_PER_BLOCK // 32,  # 256 threads → 8 warps (power of two)
    )