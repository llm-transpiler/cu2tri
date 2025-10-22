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
    Each program (grid block) handles BLOCK_SIZE threads.
    The kernel iterates over the vector with a stride equal to
    BLOCK_SIZE * grid_size, exactly reproducing the 8‑pass
    strided pattern of the CUDA version.
    """
    pid = tl.program_id(0)                     # block index (blockIdx.x)
    # Thread indices within the block
    thread_idx = tl.arange(0, BLOCK_SIZE)
    # Global offsets for the first iteration
    offsets = pid * BLOCK_SIZE + thread_idx
    # Total number of threads across the entire grid
    stride = BLOCK_SIZE * tl.num_programs(0)

    # Loop over the vector in stride‑sized chunks
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
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor sizes must be at least `size`"

    BLOCK_SIZE = 1024  # matches original CUDA block size
    GRID_SIZE = 256    # matches original CUDA grid size

    # Launch the Triton kernel
    _triton_kernel_impl[GRID_SIZE](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()