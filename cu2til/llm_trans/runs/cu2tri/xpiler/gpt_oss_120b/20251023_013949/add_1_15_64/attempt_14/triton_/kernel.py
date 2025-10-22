import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that performs element‑wise addition:
        C[i] = A[i] + B[i]   for i in [0, size)
    BLOCK_SIZE must be a power‑of‑two (required by tl.arange).
    """
    pid = tl.program_id(0)  # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original CUDA kernel launch.
    Parameters
    ----------
    A, B, C : torch.Tensor
        Tensors residing on the GPU. Any shape is accepted as long as the
        total number of elements is at least `size`. The tensors are treated
        as contiguous 1‑D buffers.
    size : int
        Number of elements to process.
    """
    # Basic validation
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("Only torch.float32 tensors are supported.")
    if size < 0:
        raise ValueError("size must be non‑negative.")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError("Tensor length is smaller than the requested size.")
    if size == 0:
        return

    # Ensure contiguous memory for Triton loads/stores
    A_contig = A if A.is_contiguous() else A.contiguous()
    B_contig = B if B.is_contiguous() else B.contiguous()
    C_contig = C if C.is_contiguous() else C.contiguous()

    BLOCK_SIZE = 1024  # power‑of‑two required by tl.arange
    grid = lambda meta: ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _triton_kernel_impl[grid](
        A_contig,
        B_contig,
        C_contig,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )