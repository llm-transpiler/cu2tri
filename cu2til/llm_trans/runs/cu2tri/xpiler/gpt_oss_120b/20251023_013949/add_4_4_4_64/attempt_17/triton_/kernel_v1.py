import torch
import triton
import triton.language as tl

# Triton kernel implementation
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # float* __restrict__ A
    B_ptr,          # float* __restrict__ B
    C_ptr,          # float* __restrict__ T_add (output)
    N,              # int size (number of elements to process)
    BLOCK_SIZE: tl.constexpr,  # compile‑time block size
):
    pid = tl.program_id(0)                     # block index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N                          # bounds check

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point wrapper matching the original CUDA kernel signature.
    Parameters:
        A (torch.Tensor): input tensor of shape (size,) and dtype torch.float32
        B (torch.Tensor): input tensor of shape (size,) and dtype torch.float32
        C (torch.Tensor): output tensor of shape (size,) and dtype torch.float32
        size (int): number of elements to process (mirrors the original `size` argument)
    """
    # Basic sanity checks (mirrors CUDA __restrict__ expectations)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"

    # Ensure the provided size does not exceed the actual tensor lengths
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), \
        "Provided size exceeds tensor dimensions"

    BLOCK_SIZE = 1024
    # Compute grid size: one program (block) per BLOCK_SIZE elements
    grid = ((size + BLOCK_SIZE - 1) // BLOCK,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
    )