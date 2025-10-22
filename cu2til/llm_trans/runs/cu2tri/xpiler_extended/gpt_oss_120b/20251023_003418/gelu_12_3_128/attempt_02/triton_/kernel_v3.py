import torch
import triton
import triton.language as tl
import math

# Compile‑time constants for the GELU approximation
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)  # ≈ 0.7978845608
COEFF = 0.044715

@triton.jit
def _triton_kernel_impl(
    A_ptr,                # *float32
    C_ptr,                # *float32
    size,                 # i32
    BLOCK_SIZE: tl.constexpr,
    SQRT_2_OVER_PI: tl.constexpr,
    COEFF: tl.constexpr,
):
    """
    Triton kernel that mirrors the original CUDA implementation.
    Applies the GELU approximation element‑wise:
        0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
    """
    pid = tl.program_id(0)                                   # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)    # thread indices
    mask = offsets < size                                     # bounds check

    # Load input (out‑of‑bounds elements are ignored)
    x = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # GELU approximation
    x_cubed = x * x * x
    inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed)
    gelu = 0.5 * x * (1.0 + tl.tanh(inner))

    # Store result
    tl.store(C_ptr + offsets, gelu, mask=mask)

def triton_kernel(A: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper matching the original CUDA kernel signature:
        cuda_kernel(float *A, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32) on CUDA device.
    C : torch.Tensor
        Output tensor (float32) on CUDA device.
    size : int
        Number of elements to process.
    """
    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    if not (A.is_cuda and C.is_cuda):
        raise AssertionError("Both tensors must reside on a CUDA device")
    if A.dtype != torch.float32 or C.dtype != torch.float32:
        raise AssertionError("Only float32 tensors are supported")
    if A.numel() < size or C.numel() < size:
        raise AssertionError("Tensor storage insufficient for the requested size")

    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    C = C.contiguous()

    # ------------------------------------------------------------------
    # Kernel launch configuration
    # ------------------------------------------------------------------
    BLOCK_SIZE = 1024  # matches __launch_bounds__(1024) in the CUDA code
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        SQRT_2_OVER_PI=SQRT_2_OVER_PI,
        COEFF=COEFF,
    )

    # Synchronize to guarantee completion before returning (useful for testing)
    torch.cuda.synchronize()