import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel that mirrors the CUDA implementation
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A: tl.tensor,          # input tensor A
    B: tl.tensor,          # input tensor B
    T_add: tl.tensor,      # output tensor (C in the original code)
    size: tl.int32,        # logical size (not used for indexing, kept for API compatibility)
    BLOCK_SIZE: tl.constexpr,
):
    # Thread-local offsets 0 .. BLOCK_SIZE-1 (identical to threadIdx.x in CUDA)
    offs = tl.arange(0, BLOCK_SIZE)

    # Guard against out‑of‑bounds when size < BLOCK_SIZE (original CUDA would be undefined)
    mask = offs < size

    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    c = a + b
    tl.store(T_add + offs, c, mask=mask)


# ----------------------------------------------------------------------
# Wrapper that matches the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that mimics the original `cuda_kernel` launch.
    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors residing on the CUDA device.
    size : int
        Logical size of the vectors (used only for grid calculation).
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"

    BLOCK_SIZE = 320  # matches __launch_bounds__(320) in the CUDA code

    # Compute number of blocks (ceil division) – identical to the CUDA launch configuration
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )