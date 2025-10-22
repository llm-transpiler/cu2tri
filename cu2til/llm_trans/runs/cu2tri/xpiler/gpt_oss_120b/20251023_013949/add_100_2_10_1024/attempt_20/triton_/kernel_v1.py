import
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
    Each thread processes a strided sequence of elements, mirroring the
    original CUDA kernel that used an outer loop of 8 iterations.
    """
    # Program (block) identifier – equivalent to CUDA's blockIdx.x
    pid = tl.program_id(0)

    # Total number of threads launched (grid * block)
    total_threads = BLOCK_SIZE * tl.num_programs(0)

    # Number of strided iterations each thread must perform to cover `size`
    num_iters = (size + total_threads - 1) // total_threads

    i = 0
    while i < num_iters:
        # Linear index for this iteration
        offsets = pid * BLOCK_SIZE + i * total_threads + tl.arange(0, BLOCK_SIZE)
        mask = offsets < size

        a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
        tl.store(C_ptr + offsets, a + b, mask=mask)

        i += 1


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point wrapper that mimics the original CUDA kernel signature.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32 on CUDA.
    size : int
        Number of elements to process.
    """
    # Sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32
    assert A.numel >= size and B.numel() >= size and C.numel() >= size

    BLOCK_SIZE = 1024   # matches the original CUDA block size
    GRID_SIZE = 256     # matches the original CUDA grid size

    # Launch the Triton kernel
    _triton_kernel_impl[GRID_SIZE](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()