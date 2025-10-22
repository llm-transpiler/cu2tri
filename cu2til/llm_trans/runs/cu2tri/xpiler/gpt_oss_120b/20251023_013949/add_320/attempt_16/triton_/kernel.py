import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Triton kernel implementing element‑wise addition.
# Must be named exactly `_triton_kernel_impl` as required.
# -------------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float32
    B_ptr,          # *float32
    T_add_ptr,      # *float32 (output)
    size,           # int32 scalar: number of elements to process
    BLOCK_SIZE: tl.constexpr,  # compile‑time constant: threads per program
):
    pid = tl.program_id(0)                     # 1‑D grid index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size                       # guard against OOB

    a = tl.load(A_ptr + offsets, mask=mask)    # load A
    b = tl.load(B_ptr + offsets, mask=mask)    # load B
    tl.store(T_add_ptr + offsets, a + b, mask=mask)  # store result


# -------------------------------------------------------------------------
# Wrapper matching the original CUDA kernel signature:
#   void cuda_kernel(float *A, float *B, float *C, int size)
# -------------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mirrors the original CUDA kernel signature.
    Performs C[i] = A[i] + B[i] for i in [0, size).

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA) with at least `size` elements.
    B : torch.Tensor
        Input tensor (float32, CUDA) with at least `size` elements.
    C : torch.Tensor
        Output tensor (float32, CUDA) with at least `size` elements.
    size : int
        Number of elements to process.
    """
    # -----------------------------------------------------------------
    # Basic validation
    # -----------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must have dtype torch.float32.")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise RuntimeError("Tensor sizes must be at least `size`.")

    # -----------------------------------------------------------------
    # Triton launch configuration
    # -----------------------------------------------------------------
    BLOCK_SIZE = 256          # 256 threads = 8 warps (power‑of‑2, satisfies Triton)
    NUM_WARPS = 8             # Must be a power of two
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)  # tuple‑style grid

    # -----------------------------------------------------------------
    # Kernel launch
    # -----------------------------------------------------------------
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )