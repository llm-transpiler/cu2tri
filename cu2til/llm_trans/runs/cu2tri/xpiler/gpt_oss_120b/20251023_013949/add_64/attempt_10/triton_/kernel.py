import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes:
        C[i] = A[i] + B[i]  for i in [0, N)
    BLOCK_SIZE is a compile‑time constant (64 in this case).
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global offsets for this block
    mask = offsets < N                         # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point wrapper that mirrors the original CUDA kernel signature.
    Performs element‑wise addition C = A + B for `size` elements.
    """
    # -------------------------------------------------------------------------
    # Input validation (mirrors expectations of the original CUDA code)
    # -------------------------------------------------------------------------
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor), \
        "All inputs must be torch.Tensor objects"
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Only float32 tensors are supported"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "Tensors must be contiguous"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), \
        "size exceeds the number of elements in one of the tensors"

    # -------------------------------------------------------------------------
    # Kernel launch configuration
    # -------------------------------------------------------------------------
    BLOCK_SIZE = 64
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # 1‑D grid

    # -------------------------------------------------------------------------
    # Launch the Triton kernel
    # -------------------------------------------------------------------------
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)