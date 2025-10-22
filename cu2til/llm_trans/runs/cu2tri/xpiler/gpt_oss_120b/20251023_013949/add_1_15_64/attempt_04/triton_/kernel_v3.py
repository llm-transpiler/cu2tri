import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    A_ptr,  # tl.pointer(tl.float32)
    B_ptr,  # tl.pointer(tl.float32)
    C_ptr,  # tl.pointer(tl.float32)
    N: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """Element‑wise addition kernel.

    Each thread computes one element. BLOCK_SIZE must be a power‑of‑2
    (required by tl.arange). We use 1024 threads per block, which
    satisfies the original launch bound of ≤960 threads while keeping
    the range power‑of‑2.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel.
    Parameters match the original CUDA signature:
        A (torch.Tensor): input tensor A (float32, CUDA)
        B (torch.Tensor): input tensor B (float32, CUDA)
        C (torch.Tensor): output tensor C (float32, CUDA)
        size (int): number of elements to process
    """
    # Basic validation
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "Tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "Tensors must be contiguous"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, "Tensor size insufficient for given size"

    # Triton requires a power‑of‑2 block size for tl.arange
    BLOCK_SIZE = 1024  # nearest power‑of‑2 ≥ 960
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch with a power‑of‑2 number of warps (32 warps = 1024 threads)
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )
    torch.cuda.synchronize()