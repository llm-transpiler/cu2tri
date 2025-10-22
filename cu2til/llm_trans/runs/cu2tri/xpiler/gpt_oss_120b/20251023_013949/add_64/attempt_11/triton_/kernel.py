import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes T_add[i] = A[i] + B[i] for i in [0, N).
    """
    pid = tl.program_id(0)                      # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices for this block
    mask = offsets < N                           # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(T_add_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel with the same semantics as the original CUDA kernel.

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors residing on the same CUDA device.
    size : int
        Number of elements to process (equivalent to the `size` argument in the CUDA host function).
    """
    # Basic sanity checks (mirrors typical CUDA expectations)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32
    assert A.dim() == B.dim() == C.dim() == 1

    # Ensure contiguous memory layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    BLOCK_SIZE = 64
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # one‑dimensional grid

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)