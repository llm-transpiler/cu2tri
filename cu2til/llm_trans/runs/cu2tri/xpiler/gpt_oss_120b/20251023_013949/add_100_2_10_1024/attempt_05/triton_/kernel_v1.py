import torch
import triton
import triton.language as tl

# Triton kernel that performs elementwise addition: C = A + B
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    A_ptr, B_ptr, C_ptr: pointers float (device memory)
    N: total number of elements to (constant 2048000 in the original CUDA kernel)
    BLOCK_SIZE: compile‑time constant, set to 1024 (matches CUDA launch bounds)
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # linear indices for this block
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A, B, C, size):
    """
    Entry‑point that mimics the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors residing on the same CUDA device.
    size : int
        Ignored – the original CUDA kernel processes a fixed 2,048,000 elements.
    """
    # Validate inputs
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32
    assert A.shape == B.shape == C.shape, "All tensors must have the same shape"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous()

    # Constants matching the original CUDA implementation
    BLOCK_SIZE = 1024
    TOTAL_ELEMENTS = 2048000  # hard‑coded bound from the CUDA kernel

    # Compute the number of program blocks needed
    num_blocks = (TOTAL_ELEMENTS + BLOCK_SIZE - 1) // BLOCK_SIZE  # = 2000
    grid = (num_blocks,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        TOTAL_ELEMENTS,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Optional: synchronize to mimic CUDA kernel launch semantics
    torch.cuda.synchronize()