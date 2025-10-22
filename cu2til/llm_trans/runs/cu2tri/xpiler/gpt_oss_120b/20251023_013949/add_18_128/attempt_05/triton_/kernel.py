import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernel implementation (must be named exactly as requested)
# ------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A,          # *float32
    B,          # *float32
    C,          # *float32
    size,       # i32
    BLOCK_SIZE: tl.constexpr,  # compile‑time constant (1024)
):
    """
    Compute C[i] = A[i] + B[i] for i < size.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices
    mask = offsets < size                       # guard out‑of‑bounds

    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, a + b, mask=mask)


# ------------------------------------------------------------------
# Wrapper function matching the original CUDA kernel signature
# ------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that mirrors the original CUDA kernel:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32) on CUDA device.
    B : torch.Tensor
        Input tensorfloat) on device.
    C : torch.Tensor        Output (float32) on CUDA device; will be overwritten.
    size : int
        Number of elements to process.
    """
    # ------------------------------------------------------------------
    # Basic sanity checks (mirrors the expectations of the CUDA code)
    # ------------------------------------------------------------------
    if not A.is_cuda or not B.is_cuda or not C.is_cuda:
        raise RuntimeError("All tensors must reside on a CUDA device.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32.")
    if A.shape != B.shape or A.shape != C.shape:
        raise RuntimeError("All tensors must have the same shape.")
    if size > A.numel():
        raise RuntimeError("Provided size exceeds tensor length.")

    # Ensure contiguous layout for optimal memory access
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # ------------------------------------------------------------------
    # Launch configuration (mirrors <<<numBlocks, blockSize>>>)
    # ------------------------------------------------------------------
    BLOCK_SIZE = 1024
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
    )