import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Global offsets for this program instance
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask out-of-range elements
    mask = offsets < size
    # Load inputs with masking (out‑of‑range loads return 0.0)
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    # Compute addition
    c = a + b
    # Store results with masking
    tl.store(C_ptr + offsets, c, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton implementation of the CUDA kernel:
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
    # Validate inputs
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be on the CUDA device")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise RuntimeError("All tensors must be of type torch.float32")
    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    # Early exit for empty tensors
    if size == 0:
        return
    BLOCK_SIZE = 1024
    # Compute 1‑D grid size exactly as the original CUDA launch
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)