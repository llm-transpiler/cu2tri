import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes C = A + B elementwise for `size` elements.
    Mirrors the original CUDA kernel which launched 256 blocks of 1024 threads
    each and performed an inner loop of 8 iterations.
    """
    pid = tl.program_id(0)                     # blockIdx.x
    base_offset = pid * BLOCK_SIZE             # blockIdx.x * 1024
    offsets = tl.arange(0, BLOCK_SIZE, dtype=tl.int32)  # threadIdx.x

    # The original CUDA kernel iterated 8 times, each time adding a stride of
    # 256 * BLOCK_SIZE (262144) to the index.
    for outer in range(8):
        index = outer * 256 * BLOCK_SIZE + base_offset + offsets
        mask = index < size
        a = tl.load(A_ptr + index, mask=mask, other=0.0)
        b = tl.load(B_ptr + index, mask=mask, other=0.0)
        tl.store(C_ptr + index, a + b, mask=mask)

def triton_kernel(A, B, C, size):
    """
    Entry‑point that launches the Triton kernel.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) on CUDA device, dtype torch.float32.
    B : torch.Tensor
        Input tensor of shape (size,) on CUDA device, dtype torch.float32.
    C : torch.Tensor
        Output tensor of shape (size,) on CUDA device, dtype torch.float32.
    size : int
        Number of elements to process.
    """
    # Validate inputs (mirrors the expectations of the original CUDA kernel)
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)
    assert A.is_cuda and B.is_cuda and C.is_cuda
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32

    BLOCK_SIZE = 1024
    # Original launch: 256 blocks, each with 1024 threads    grid = (256,)
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)