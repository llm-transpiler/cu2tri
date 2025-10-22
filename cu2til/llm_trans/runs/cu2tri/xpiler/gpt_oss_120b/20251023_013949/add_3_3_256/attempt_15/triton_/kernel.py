import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes C = A + B element‑wise for `size` elements.
    Mirrors the original CUDA kernel that used 1024 threads per block.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size                       # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that launches the Triton kernel.
    Parameters
    ----------
    A, B, C : torch.Tensor
        Input tensors A and B, and output tensor C. All must be
        contiguous, on the same CUDA device, and of dtype torch.float32.
    size : int
        Number of elements to process (must be ≤ number of elements in the tensors).
    """
    # ------------------------------------------------------------------
    # Input validation (mirrors the expectations of the original CUDA code)
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("All tensors must be of dtype torch.float32.")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()
    if size > A.numel() or size > B.numel() or size > C.numel():
        raise ValueError("`size` exceeds the number of elements in the tensors.")

    BLOCK_SIZE = 1024                     # matches the original CUDA block size
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # one‑dimensional grid

    # Launch the kernel. `num_warps` is chosen for good occupancy on Hopper GPUs.
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )