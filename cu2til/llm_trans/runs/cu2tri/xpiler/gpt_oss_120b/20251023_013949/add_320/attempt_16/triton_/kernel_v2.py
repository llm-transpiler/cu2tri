import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise addition.
# Must be named exactly `_triton_kernel_impl`.
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float32
    B_ptr,          # *float32
    T_add_ptr,      # *float32
    size,           # int32 scalar
    BLOCK_SIZE: tl.constexpr  # compile‑time constant: threads per program
):
    pid = tl.program_id(0)                     # 1‑D grid index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size                       # Guard against OOB

    a = tl.load(A_ptr + offsets, mask=mask)    # Load A
    b = tl.load(B_ptr + offsets, mask=mask)    # Load B
    tl.store(T_add_ptr + offsets, a + b, mask=mask)  # Store result


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point mirroring the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA)
    B : torch.Tensor
        Input tensor (float32, CUDA)
    C : torch.Tensor
        Output tensor (float32, CUDA)
    size : int
        Number of elements to process
    """
    # Basic sanity checks
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must have dtype torch.float32.")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise RuntimeError("Tensor sizes must be at least `size`.")

    # Triton launch configuration
    BLOCK_SIZE = 256          # 256 threads = 8 warps (power‑of‑2, satisfies Triton)
    NUM_WARPS = 8             # Must be a power of two
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )