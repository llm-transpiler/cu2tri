import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementation (named exactly as required)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr, B_ptr, C_ptr,          # pointers to the input / output buffers
    N,                            # total number of elements to process
    BLOCK_SIZE: tl.constexpr      # compile‑time constant: threads per block (must be power of 2)
):
    """
    Element‑wise addition: C = A + B
    Mirrors the original CUDA kernel that used 960 threads per block.
    """
    pid = tl.program_id(0)                     # 1‑D grid indexing
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard against out‑of‑bounds accesses (when size is not a multiple of BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    tl.store(C_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that replicates the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D contiguous tensors on the current CUDA device, dtype=torch.float32.
    size : int
        Number of elements to process (may be <= A.numel()).
    """
    # ------------------------------------------------------------------
    # Basic sanity checks (mirroring the expectations of the CUDA code)
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be on CUDA")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("All tensors must be float32")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise RuntimeError("All tensors must be contiguous")
    if size < 0:
        raise ValueError("size must be non‑negative")
    if size == 0:
        return  # nothing to do
    if size > A.numel() or size > B.numel() or size > C.numel():
        raise ValueError("size exceeds tensor length")

    # ------------------------------------------------------------------
    # Configuration (matches the original launch bounds)
    # ------------------------------------------------------------------
    BLOCK_SIZE = 1024  # power‑of‑two block size to satisfy triton's arange requirement
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # 1‑D grid

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,   # 1024 threads per block → 32 warps
        num_stages=2    # modest pipeline depth for good occupancy on H800
    )