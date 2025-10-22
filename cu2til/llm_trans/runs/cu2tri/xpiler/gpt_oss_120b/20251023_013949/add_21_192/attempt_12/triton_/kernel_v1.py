import torch
import triton
import triton.language as tl

# Triton kernel implementation (named as required)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_ptr,
                        BLOCK_SIZE: tl.constexpr,
                        BOUND: tl.constexpr):
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices within the block
    mask = offsets < BOUND                     # emulate the CUDA bound check (4032)
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(T_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor,
                  B: torch.Tensor,
                  C: torch.Tensor,
                  size: int):
    """
    Wrapper that mimics the original CUDA kernel signature.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32) on CUDA device.
    B : torch.Tensor
        Input tensor (float32) on CUDA device.
    C : torch.Tensor
        Output tensor (float32) on CUDA device.
    size : int
        Logical size of the vectors (used only for grid calculation, identical to CUDA wrapper).
    """
    # Basic sanity checks (mirroring typical CUDA expectations)
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must reside on a CUDA device")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("All tensors must be torch.float32")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise RuntimeError("All tensors must be contiguous")

    BLOCK_SIZE = 1024
    # Compute grid size based on the provided logical size (same as CUDA wrapper)
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        BLOCK_SIZE=BLOCK_SIZE,
        BOUND=4032,          # hard‑coded bound from the original CUDA kernel
        num_warps=8          # 1024 threads ≈ 32 warps; 8 warps gives good occupancy on modern GPUs
    )