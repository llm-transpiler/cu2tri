import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Linear offsets for each thread in the block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Hard‑coded bound from the original CUDA kernel (2304)
    mask = offsets < 2304
    # Load with mask (out‑of‑bounds loads return 0.0)
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    # Compute addition
    c = a + b
    # Store result with mask
    tl.store(T_add_ptr + offsets, c, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton equivalent of the CUDA kernel:
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
        Logical size of the vectors (used only for grid calculation)
    """
    # Sanity checks
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must reside on the CUDA device")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise RuntimeError("All tensors must be of type torch.float32")
    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    BLOCK_SIZE = 1024
    # Compute grid size exactly as the original CUDA launch
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE=BLOCK_SIZE)