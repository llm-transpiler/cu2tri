import torch
import triton
import triton.language as tl

# Triton kernel implementing the element‑wise addition with bounds checking
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float32
    B_ptr,          # *float32
    C_ptr,          # *float32
    size,           # int64
    BLOCK_SIZE: tl.constexpr  # compile‑time constant matching CUDA launch bounds
):
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices within the grid
    mask = offsets < size                       # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton wrapper that mirrors the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA, contiguous)
    B : torch.Tensor
        Input tensor B (float32, CUDA, contiguous)
    C : torch.Tensor
        Output tensor C (float32, CUDA, contiguous)
    size : int
        Number of elements to process (must be ≤ A.numel(), B.numel(), C.numel())
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "Tensors must be contiguous"
    assert A.device == B.device == C.device, "All tensors must reside on the same device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "Tensors must be float32"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), "size exceeds tensor length"

    BLOCK_SIZE = 1024  # matches __launch_bounds__(1024) in the CUDA implementation
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8  # Adjust for optimal occupancy on the target GPU
    )