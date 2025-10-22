import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel (compile‑time constant BLOCK_SIZE)
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024  # threads per block, matches the CUDA launch bounds

@triton.jit
def _triton_kernel_impl(A, B, C, size, BLOCK_SIZE: tl.constexpr):
    """
    Element‑wise addition: C[i] = A[i] + B[i] for i < size.
    Mirrors the behavior of the original CUDA kernel.
    """
    # Program (block) identifier
    pid = tl.program_id(axis=0)                # blockIdx.x

    # Compute the absolute offset of each thread within the grid
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # threadIdx.x

    # Guard against out‑of‑bounds accesses
    mask = offsets < size

    # Load inputs (masked)
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)

    # Store the result (masked)
    tl.store(C + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper that mimics the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point with the same signature as the CUDA kernel:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA)
    B : torch.Tensor
        Input tensor B (float32, CUDA)
    C : torch.Tensor
        Output tensor C (float32, CUDA)
    size : int
        Number of elements to process
    """
    # ------------------------------------------------------------------
    # Validation (mirrors typical CUDA expectations)
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must reside on the CUDA device")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("All tensors must be torch.float32")
    if size < 0:
        raise ValueError("size must be non‑negative")

    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # ------------------------------------------------------------------
    # Grid configuration: one‑dimensional grid matching CUDA launch
    # ------------------------------------------------------------------
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)