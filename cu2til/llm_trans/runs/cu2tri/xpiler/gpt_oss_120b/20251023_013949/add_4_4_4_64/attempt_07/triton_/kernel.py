import torch
import triton
import triton.language as tl

# Triton kernel implementing element‑wise addition
@triton.jit
def _triton_kernel_impl(
    A_ptr,  # pointer to float32
    B_ptr,  # pointer to float32
    C_ptr,  # pointer to float32
    size,   # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global offsets
    mask = offsets < size                       # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask)    # load A[i] where valid
    b = tl.load(B_ptr + offsets, mask=mask)    # load B[i] where valid
    tl.store(C_ptr + offsets, a + b, mask=mask)  # store result


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton implementation of the original CUDA kernel.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32 on CUDA.
    size : int
        Number of elements to process (must be <= A.numel()).
    """
    # Basic sanity checks – mirrors the expectations of the CUDA code
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), \
        "size must not exceed tensor lengths"

    BLOCK_SIZE = 1024  # matches the original CUDA launch bounds
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Optional synchronization (uncomment if needed)
    # torch.cuda.synchronize()