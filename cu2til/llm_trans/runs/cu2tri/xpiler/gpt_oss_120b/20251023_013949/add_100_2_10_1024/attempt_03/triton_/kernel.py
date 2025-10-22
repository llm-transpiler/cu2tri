import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(T_add_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton entry point that mimics the original CUDA kernel signature.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, contiguous, on CUDA).
    B : torch.Tensor
        Input tensor B (float32, contiguous, on CUDA).
    C : torch.Tensor
        Output tensor C (float32, contiguous, on CUDA).
    size : int
        Number of elements to process (must be <= A.numel()).
    """
    # Validate inputs
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "All tensors must be contiguous"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), "size exceeds tensor size"

    BLOCK_SIZE = 1024
    # Compute grid size (number of program instances)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()