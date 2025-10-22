import torch
import triton
import triton.language as tl

# Triton kernel implementing the same logic as the original CUDA kernel
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # float* __restrict__ A
    B_ptr,               # float* __restrict__ B
    T_add_ptr,           # float* __restrict__ T_add (output)
    N: tl.constexpr,     # compile‑time constant bound (2304 in the CUDA code)
    BLOCK_SIZE: tl.constexpr  # compile‑time constant block size (1024)
):
    # program id corresponds to blockIdx.x in CUDA
    pid = tl.program_id(0)
    # Global offsets for this block
    offs = pid * BLOCK_SIZE tl.arange(, BLOCK_SIZE)
    # Fixed bound check: only process indices < 2304
    mask = offs < N

    # Load values with masking (out‑of‑range loads yield 0.0)
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # Store the sum, respecting the mask
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point mirroring the original `cuda_kernel` signature.
    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors on the same CUDA device.
    size : int
        Logical vector length (used only to compute grid dimensions,
        exactly as the original CUDA wrapper did).
    """
    # Basic validation
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must reside on a CUDA device")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32")
    if A.shape != B.shape or A.shape != C.shape:
        raise RuntimeError("All tensors must have the same shape")
    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    BLOCK_SIZE = 1024   # matches __launch_bounds__(1024) in the CUDA kernel
    N = 2304            # fixed bound from the original kernel's if‑statement

    # Compute grid size exactly like the CUDA wrapper
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        N,
        BLOCK_SIZE,
        # Launch‑time configuration; 8 warps (256 threads) works well for 1024‑element blocks
        num_warps=8,
        num_stages=3,
    )
    # No explicit synchronization needed; callers may synchronize if required.