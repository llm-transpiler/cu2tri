import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Triton kernel: element‑wise addition (mirrors the original CUDA kernel)
# -------------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float32
    B_ptr,          # *float32
    C_ptr,          # *float32
    size: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute C = A + B for indices < size.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices

    # Guard against out‑of‑bounds accesses
    mask = offsets < size

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    tl.store(C_ptr + offsets, a + b, mask=mask)


# -------------------------------------------------------------------------
# Wrapper that matches the original CUDA kernel signature
# -------------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that replicates the CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA) of length >= size.
    B : torch.Tensor
        Input tensor (float32, CUDA) of length >= size.
    C : torch.Tensor
        Output tensor (float32, CUDA) of length >= size.
    size : int
        Number of elements to process.
    """
    # -----------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must have dtype torch.float32.")
    if A.shape != B.shape or A.shape != C.shape:
        raise RuntimeError("All tensors must have the same shape.")
    if size < 0 or size > A.numel():
        raise ValueError("Invalid size argument for the provided tensors.")

    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # -----------------------------------------------------------------
    # Triton launch configuration (mirrors CUDA launch bounds)
    # -----------------------------------------------------------------
    BLOCK_SIZE = 1024  # matches __launch_bounds__(1024) in the CUDA code
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # -----------------------------------------------------------------
    # Kernel launch
    # -----------------------------------------------------------------
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,          # tuned for 1024‑thread blocks on modern GPUs
    )
    # Synchronize to make the launch observable from host code
    torch.cuda.synchronize()