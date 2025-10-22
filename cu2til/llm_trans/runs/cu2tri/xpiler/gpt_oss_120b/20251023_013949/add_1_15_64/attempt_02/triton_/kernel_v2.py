import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N,
                        BLOCK_SIZE: tl.constexpr):
    """Element‑wise addition kernel: C[i] = A[i] + B[i] for i < N."""
    pid = tl.program_id(0)  # block index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton wrapper mirroring the original CUDA kernel signature:
    void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A, B, C : torch.Tensor
        Contiguous, float32 tensors residing on the same CUDA device.
    size : int
        Number of elements to process (must not exceed tensor lengths).
    """
    # Input validation
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise ValueError("All tensors must be contiguous.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("All tensors must be of type torch.float32.")
    if A.device != B.device or A.device != C.device:
        raise ValueError("All tensors must be on the same device.")
    if size > A.numel() or size > B.numel() or size > C.numel():
        raise ValueError("`size` exceeds one or more tensor lengths.")
    # Triton launch configuration
    BLOCK_SIZE = 1024  # power‑of‑two for tl.arange
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    # Kernel launch
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE
    )