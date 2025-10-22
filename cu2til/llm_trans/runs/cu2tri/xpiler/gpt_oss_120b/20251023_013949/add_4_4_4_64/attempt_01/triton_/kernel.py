import torch
import triton
import triton.language as tl

# Block size matches the CUDA launch configuration (1024 threads per block)
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float32
    B_ptr,          # *float32
    C_ptr,          # *float32
    size,           # i32
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton implementation of the element‑wise addition kernel.
    Mirrors the behavior of the original CUDA kernel:
        if (global_idx < size) C[global_idx] = A[global_idx] + B[global_idx];
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices
    mask = offsets < size                       # guard out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that matches the original CUDA kernel signature.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor ``float32`` on CUDA device.
    B : torch.Tensor
        Input tensor ``float32`` on CUDA device.
    C : torch.Tensor
        Output tensor ``float32`` on CUDA device.
    size : int
        Number of elements to process (must be ≤ A.numel(), B.numel(), C.numel()).
    """
    # Basic validation – mirrors the expectations of the CUDA code
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), \
        "size exceeds tensor length"

    # Ensure contiguous memory layout for optimal Triton access
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # Compute grid dimensions (equivalent to CUDA's numBlocks)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )