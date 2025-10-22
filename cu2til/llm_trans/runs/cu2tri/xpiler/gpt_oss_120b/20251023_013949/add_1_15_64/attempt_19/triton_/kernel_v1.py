import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementation (named exactly as required)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Element‑wise addition: C = A + B
    Parameters
    ----------
    A_ptr : pointer to float32 (input)
    B_ptr : pointer to float32 (input)
    C_ptr : pointer to float32 (output)
    N     : total number of elements to process
    BLOCK_SIZE : compile‑time constant (set to 960)
    """
    pid = tl.program_id(0)                     # block index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)

# ----------------------------------------------------------------------
# Wrapper function matching the original CUDA signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Launches the Triton kernel.
    Mirrors the signature of the original CUDA kernel:
        cuda_kernel(float *A, float *B, float *C, int size)
    """
    # ------------------------------------------------------------------
    # Basic sanity checks (mirroring typical CUDA expectations)
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise RuntimeError("Tensor sizes must be at least 'size'")

    # Ensure contiguous memory layout for optimal pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    BLOCK_SIZE = 960
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )   # 1‑D grid
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

    # Optional: synchronize for debugging (comment out in production)
    # torch.cuda.synchronize()