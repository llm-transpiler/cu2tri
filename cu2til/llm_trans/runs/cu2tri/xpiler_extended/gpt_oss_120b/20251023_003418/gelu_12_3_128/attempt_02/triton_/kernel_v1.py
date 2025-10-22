import torch
import triton
import triton.language as tl
import math

# Constants for the GELU approximation
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)  # sqrt(2/pi)
COEFF = 0.044715

@triton.jit
def _triton_kernel_impl(A_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton implementation of the CUDA kernel that applies the GELU
    approximation element‑wise to the input tensor A and writes the result
    to tensor C.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices within the block
    mask = offsets < size                       # bounds check

    # Load input values (out‑of‑bounds elements are set to 0.0 and ignored later)
    x = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_cubed = x * x * x
    inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed)
    gelu = 0.5 * x * (1.0 + tl.tanh(inner))

    # Write results back
    tl.store(C_ptr + offsets, gelu, mask=mask)

def triton_kernel(A: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA kernel signature:
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
    # Basic sanity checks
    assert A.is_cuda and C.is_cuda, "Tensors must reside on a CUDA device"
    assert A.dtype == torch.float32 and C.dtype == torch.float32, "Only float32 tensors are"
    assert.numel >= and.num() >= size, "Tensor storage insufficient for the requested size"

    BLOCK_SIZE = 1024  # matches __launch_bounds__(1024) in the CUDA code
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # 1‑D grid

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, C, size, BLOCK_SIZE=BLOCK_SIZE)