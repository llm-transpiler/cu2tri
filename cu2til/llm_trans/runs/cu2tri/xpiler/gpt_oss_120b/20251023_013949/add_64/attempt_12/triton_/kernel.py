import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr):
    # Program (block) ID – kept for parity with CUDA launch but not used for indexing.
    pid = tl.program_id(0)
    # Thread index inside the block (0‑63). Mirrors `threadIdx.x` in the CUDA kernel.
    offs = tl.arange(0, 64)

    # Load the two input values, compute the sum, and store the result.
    a = tl.load(A_ptr + offs)
    b = tl.load(B_ptr + offs)
    tl.store(T_add_ptr + offs, a + b)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton equivalent of the original CUDA kernel `cuda_kernel`.
    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D CUDA tensors of dtype torch.float32.
    size : int
        Number of elements the original launch intends to cover.
    """
    # Basic sanity checks – mirrors expectations of the CUDA launch.
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the GPU"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type float32"
    assert A.ndim == 1 and B.ndim == 1 and C.ndim == 1, "All tensors must be 1‑D"
    # The original kernel ignores `size` when indexing, but we still require the buffers
    # to be at least `size` elements long to avoid illegal memory accesses.
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Input tensors must contain at least `size` elements"

    BLOCK = 64
    num_blocks = (size + BLOCK - 1) // BLOCK
    grid = (num_blocks,)

    # Launch the Triton kernel. `num_warps=2` yields 64 threads per program (2 warps).
    _triton_kernel_impl[grid](A, B, C, num_warps=2)