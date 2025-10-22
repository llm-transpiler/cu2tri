import torch
import triton
import triton.language as tl

# Triton kernel implementing the original CUDA behavior
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # *float32
    B_ptr,               # *float32
    T_add_ptr,           # *float32
    size,                # int32 (kept for signature compatibility)
    BLOCK_SIZE: tl.constexpr  # compile‑time constant: threads per block
):
    """
    Mirrors the CUDA kernel:
        - Each thread processes one element.
        - Guard against out‑of‑bounds using the hard‑coded limit 4032.
    """
    pid = tl.program_id(0)  # block index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread offsets within block
    mask = offs < 4032  # hard‑coded bound from the original CUDA kernel

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int) -> None:
    """
    Wrapper that launches the Triton kernel.
    Parameters match the original CUDA entry point:
        A, B, C : torch tensors of dtype torch.float32 on the same CUDA device.
        size   : total number of elements (used only for grid calculation).
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"

    BLOCK_SIZE = 1024  # matches __launch_bounds__(1024) in the CUDA kernel
    # Compute grid size (number of blocks) the same way the CUDA host code does
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_tuple = (grid,)  # Triton expects a tuple for grid dimensions

    # Launch the kernel
    _triton_kernel_impl[grid_tuple](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,          # 1024 threads = 32 warps
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()