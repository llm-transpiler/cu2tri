import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise addition.
# The kernel name must be exactly `_triton_kernel_impl`.
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # float* __restrict__ A
    B_ptr,               # float* __restrict__ B
    C_ptr,               # float* __restrict__ T_add (output)
    size,                # total number of elements to process
    BLOCK_SIZE: tl.constexpr,   # compile‑time block size (threads per program)
):
    """
    Each Triton program (block) processes BLOCK_SIZE contiguous elements.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # global offsets
    mask = offsets < size                       # guard against OOB

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA `cuda_kernel` signature.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA).
    B : torch.Tensor
        Input tensor B (float32, CUDA).
    C : torch.Tensor
        Output tensor C (float32, CUDA) where the result is stored.
    size : int
        Number of elements to process.
    """
    # Basic sanity checks – these are cheap and help catch misuse early.
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must reside on the CUDA device.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32.")
    if size < 0:
        raise ValueError("size must be non‑negative.")

    BLOCK_SIZE = 320                     # matches __launch_bounds__(320) in CUDA
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )   # 1‑D grid

    # Launch the Triton kernel.  The `num_warps` hint is optional; 10 warps = 320 threads.
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE, num_warps=10)