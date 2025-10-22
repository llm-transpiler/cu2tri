import torch
import triton
import triton.language as tl

# Triton kernel implementing the same elementwise addition as the CUDA kernel.
# The kernel processes indices < 4096, matching the hard‑coded bound in the original code.
@triton.jit(num_warps=32)  # 32 warps × 32 threads = 1024 threads per program
def _triton_kernel_impl(
    A,                     # pointer to float32
    B,                     # pointer to float32
    C,                     # pointer to float32 (output)
    size,                  # int32 – kept for API compatibility, not used in the kernel
    BLOCK_SIZE: tl.constexpr  # compile‑time constant, set to 1024
):
    # Program (block) identifier
    pid = tl.program_id(0)
    # Linear offsets for the current program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Hard‑coded guard identical to the CUDA `if (idx < 4096)` condition
    mask = offsets < 4096

    # Load A and B with the mask; out‑of‑range elements are ignored (other=0.0)
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)

    # Store the sum into C where the mask is true
    tl.store(C + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point wrapper that mimics the original CUDA launch configuration.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32 on CUDA.
    size : int
        Number of elements (used only for grid sizing, matching the CUDA wrapper).
    """
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be torch.float32")
    if not isinstance(size, int):
        raise TypeError("size must be an integer")

    BLOCK_SIZE = 1024
    # Compute grid size exactly as in the CUDA wrapper: ceil(size / BLOCK_SIZE)
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
    )