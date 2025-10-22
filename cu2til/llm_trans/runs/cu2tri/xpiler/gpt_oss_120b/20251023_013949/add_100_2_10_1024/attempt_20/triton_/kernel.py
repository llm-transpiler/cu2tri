import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float32
    B_ptr,          # *float32
    C_ptr,          # *float32
    size,           # int32 (runtime)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton implementation of the original CUDA vector‑add kernel.
    Each program (grid block) processes BLOCK_SIZE elements and then
    strides across the vector with a stride equal to BLOCK_SIZE * grid_size.
    """
    pid = tl.program_id(0)                     # block index (blockIdx.x)
    thread_idx = tl.arange(0, BLOCK_SIZE)      # thread indices within the block
    offsets = pid * BLOCK_SIZE + thread_idx    # base global indices for this block
    stride = BLOCK_SIZE * tl.num_programs(0)   # total number of threads across the grid

    i = 0
    while i < size:
        idx = offsets + i
        mask = idx < size
        a = tl.load(A_ptr + idx, mask=mask, other=0.0)
        b = tl.load(B_ptr + idx, mask=mask, other=0.0)
        tl.store(C_ptr + idx, a + b, mask=mask)
        i += stride


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mirrors the original CUDA kernel signature.
    Launches the Triton kernel with the same grid/block configuration.
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA"
    assert (
        A.dtype == torch.float32
        and B.dtype == torch.float32
        and C.dtype == torch.float32
    ), "All tensors must be torch.float32"
    assert (
        A.numel() >= size and B.numel() >= size and C.numel() >= size
    ), "Tensor sizes must be at least `size`"

    BLOCK_SIZE = 1024  # matches original CUDA block size
    GRID_SIZE = 256    # matches original CUDA grid size

    # Launch the Triton kernel
    _triton_kernel_impl[(GRID_SIZE,)](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()