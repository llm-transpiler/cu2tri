import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: element‑wise addition (C = A + B)
# ----------------------------------------------------------------------
BLOCK_SIZE = 64  # matches the CUDA launch bounds

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size,
                        BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that adds two vectors.

    Parameters
    ----------
    A_ptr, B_ptr, C_ptr : pointers to float32 data (device memory)
    size                : total number of elements to process
    BLOCK_SIZE          : compile‑time constant, number of threads per block
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices
    mask = offsets < size                       # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper that mimics the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A, B, C, size):
    """
    Entry‑point that launches the Triton kernel.

    Mirrors the CUDA signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor (CUDA, float32, shape >= (size,))
    B : torch.Tensor (CUDA, float32, shape >= (size,))
    C : torch.Tensor (CUDA, float32, shape >= (size,))
    size : int
        Number of elements to process.
    """
    # ------------------------------------------------------------------
    # Basic validation (mirrors typical CUDA expectations)
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32.")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise RuntimeError("Tensor storage must be at least 'size' elements.")

    # ------------------------------------------------------------------
    # Grid configuration: one block per BLOCK_SIZE elements
    # ------------------------------------------------------------------
    grid = lambda meta: ((size + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

    # Ensure kernel completion before returning (optional but convenient)
    torch.cuda.synchronize()