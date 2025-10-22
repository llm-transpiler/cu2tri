import torch
import triton
import triton.language as tl

# Constants matching the original CUDA launch configuration
BLOCK_SIZE = 1024          # threads per block (program)
MAX_ELEMENTS = 4032        # hard‑coded bound from the CUDA kernel

@triton.jit
def _triton_kernel_impl(
    A_ptr, B_ptr, T_add_ptr, size,
    BLOCK_SIZE: tl.constexpr,
    MAX_ELEMENTS: tl.constexpr
):
    """
    Triton implementation of the element‑wise addition kernel.
    Mirrors the behaviour of the original CUDA kernel:
        if (global_idx < size && global_idx < 4032)
            T_add[global_idx] = A[global_idx] + B[global_idx];
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices

    # Apply the two bounds from the CUDA code
    mask = (offsets < size) & (offsets < MAX_ELEMENTS)

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(T_add_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch, B: torch.Tensor, C: torch.Tensor, size: int    """
    Entry‑point that mim the original `cuda_kernel` signature.
   
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors residing on the same CUDA device.
    size : int
        Logical length of the vectors (the CUDA code launches enough blocks
        to cover `size` elements, but the kernel also caps work at 4032).
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32
    assert A.dim() == B.dim() == C.dim() == 1, "Only 1‑D tensors are supported"

    # Compute grid size exactly as the CUDA host code does
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[(num_blocks,)](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_ELEMENTS=MAX_ELEMENTS,
    )