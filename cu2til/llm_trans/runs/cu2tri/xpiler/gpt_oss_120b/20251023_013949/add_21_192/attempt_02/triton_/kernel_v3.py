import torch
import triton
import triton.language as tl

# Number of threads per block (matches the CUDA configuration)
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that performs element‑wise addition:
        if idx < size: C[idx] = A[idx] + B[idx]
    """
    pid = tl.program_id(0)  # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices within the block
    mask = offsets < size  # bounds check

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA kernel signature.
    Parameters
    ----------
    A, B, C : torch.Tensor
        Input and output tensors (must be float32 and reside on the same CUDA device).
    size : int
        Logical length of the vectors (identical to the `size` argument in the CUDA wrapper).
    """
    # Basic validation
    assert isinstance(size, int) and size >= 0, "size must be a non‑negative integer"
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), \
        "size must not exceed the number of elements in the tensors"

    # Compute grid size: one program instance per BLOCK_SIZE elements
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)