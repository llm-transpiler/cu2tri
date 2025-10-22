import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton implementation of elementwise addition.
    Parameters:
        A (torch.Tensor): Input tensor A (float32, CUDA)
        B (torch.Tensor): Input tensor B (float32, CUDA)
        C (torch.Tensor): Output tensor C (float32, CUDA)
        size (int): Number of elements to process
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "All tensors must be contiguous"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, "Tensor size insufficient for given size"

    BLOCK_SIZE = 960
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel with a power‑of‑2 number of warps (32)
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32
    )
    torch.cuda.synchronize()