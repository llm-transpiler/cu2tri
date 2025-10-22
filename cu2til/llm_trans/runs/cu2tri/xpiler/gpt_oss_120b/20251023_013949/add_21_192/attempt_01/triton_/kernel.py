import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = pid * BLOCK_SIZE + offsets

    mask = idx < 4032

    a = tl.load(A_ptr + idx, mask=mask, other=0.0)
    b = tl.load(B_ptr + idx, mask=mask, other=0.0)

    tl.store(T_add_ptr + idx, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton wrapper replicating the behavior of the original CUDA kernel.
    Parameters:
        A (torch.Tensor): Input tensor (float32) on CUDA device.
        B (torch.Tensor): Input tensor (float32) on CUDA device.
        C (torch.Tensor): Output tensor (float32) on CUDA device.
        size (int): Number of elements (used only for grid configuration).
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32

    # Ensure contiguous memory layout
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    BLOCK_SIZE = 1024
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )