import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing the same element‑wise addition as the CUDA kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Each program (i.e. each block) processes BLOCK_SIZE elements.
    The kernel mirrors the CUDA behavior:
        C[i] = A[i] + B[i]   for i in [0, size)
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices
    mask = offsets < size                       # guard for tail elements

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper with the exact signature of the original CUDA entry point
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Launches the Triton kernel. Arguments must match the CUDA kernel:
        A, B, C : torch.float32 CUDA tensors
        size   : number of elements to process
    """
    # ------------------------------------------------------------------
    # Input validation (mirrors typical CUDA expectations)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"

    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # ------------------------------------------------------------------
    # Grid configuration – identical to the CUDA launch parameters
    # ------------------------------------------------------------------
    BLOCK_SIZE = 320  # matches __launch_bounds__(320) in the CUDA kernel
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # 1‑D grid

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,   # power of two (>= BLOCK_SIZE/32) to satisfy Triton constraints
    )