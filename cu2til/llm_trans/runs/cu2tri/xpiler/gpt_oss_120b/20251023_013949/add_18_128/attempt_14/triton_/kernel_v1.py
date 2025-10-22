import torch
import triton
import triton.language as tl

# Block size matches the CUDA launch bounds (1024 threads per block)
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that mirrors the original CUDA kernel.
    It adds A and B element‑wise and writes the result to C,
    but only for indices < 2304 (the same hard‑coded bound in the CUDA code).
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global thread indices

    # Hard‑coded bound check (identical to the CUDA kernel)
    mask = offsets < 2304

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original `cuda_kernel` wrapper.
    Parameters:
        A, B, C : torch.cuda.FloatTensor (contiguous, float32)
        size   : total number of elements the caller intends to process
    """
    # Basic sanity checks – match the expectations of the CUDA version
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type float32"
    # Ensure contiguous memory layout
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # Grid configuration mirrors the CUDA launch (size rounded up to BLOCK_SIZE)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel.
    # `num_warps=32` yields 32 warps × 32 threads = 1024 threads per block,
    # matching the CUDA launch bounds.
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )