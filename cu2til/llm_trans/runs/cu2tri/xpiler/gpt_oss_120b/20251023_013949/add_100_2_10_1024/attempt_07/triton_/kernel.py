import torch
import triton
import triton.language as tl

# Compile‑time constant: number of elements each thread processes.
MAX_ITERS = tl.constexpr(8)

@triton.jit
def _triton_kernel_impl(
    A_ptr,               # float* __restrict__
    B_ptr,               # float* __restrict__
    C_ptr,               # float* __restrict__
    size,                # total number of elements (int)
    BLOCK_SIZE: tl.constexpr,  # threads per block (compile‑time)
):
    """Element‑wise addition kernel matching the original CUDA implementation."""
    pid = tl.program_id(0)                         # blockIdx.x
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # threadIdx.x within the block
    stride = BLOCK_SIZE * tl.num_programs(0)        # total threads across the grid

    # Each thread processes up to MAX_ITERS elements spaced by `stride`.
    for i in range(MAX_ITERS):
        idx = offsets + i * stride                  # global index for this iteration
        mask = idx < size                           # guard against out‑of‑bounds
        a = tl.load(A_ptr + idx, mask=mask, other=0.0)
        b = tl.load(B_ptr + idx, mask=mask, other=0.0)
        tl.store(C_ptr + idx, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton entry point mirroring the original CUDA kernel signature.
    A, B, C must be CUDA tensors of dtype torch.float32.
    `size` is the number of elements to process.
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Tensors must be of type torch.float32"

    BLOCK_SIZE = 1024  # Matches the CUDA block size

    # Compute a grid that guarantees coverage of `size` elements.
    # Each thread handles up to MAX_ITERS elements, so the total work per block is BLOCK_SIZE * MAX_ITERS.
    grid = ((size + BLOCK_SIZE * MAX_ITERS - 1) // (BLOCK_SIZE * MAX_ITERS),)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)