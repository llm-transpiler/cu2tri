import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    A_ptr,  # *float32
    B_ptr,  # *float32
    C_ptr,  # *float32
    size,   # int32
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that adds two vectors element‑wise.
    Mirrors the original CUDA kernel which processed the data in
    8 strided passes (total stride = BLOCK_SIZE * grid_size).
    """
    pid = tl.program_id(0)                     # block index (equivalent to blockIdx.x)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # thread indices within the block
    stride = BLOCK_SIZE * tl.num_programs(0)    # total number of threads across the grid

    # Loop over strided chunks to cover the full vector length
    for i in range(0, size, stride):
        idx = offsets + i
        mask = idx < size
        a = tl.load(A_ptr + idx, mask=mask, other=0.0)
        b = tl.load(B_ptr + idx, mask=mask, other=0.0)
        tl.store(C_ptr + idx, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA kernel signature.
    Launches the Triton kernel with the same grid/block configuration.
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor sizes must be at least `size`"

    BLOCK_SIZE = 1024  # matches original CUDA block size
    GRID_SIZE = 256    # matches original CUDA grid size

    # Launch the Triton kernel
    _triton_kernel_impl[GRID_SIZE](
        A, B, C, size,
        BLOCK_SIZEBLOCK_SIZE    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()