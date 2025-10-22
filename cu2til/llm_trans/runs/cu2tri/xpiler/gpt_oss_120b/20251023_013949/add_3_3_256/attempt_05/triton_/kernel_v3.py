import torch
import triton
import triton.language as tl

# Compile‑time constants matching the original CUDA kernel
BLOCK_SIZE = 1024  # __launch_bounds__(1024)
MAX_IDX = 2304     # Hard‑coded bound used inside the kernel

@triton.jit
def _triton_kernel_impl(
    A_ptr,               # float* __restrict__
    B_ptr,               # float* __restrict__
    T_add_ptr,           # float* __restrict__
    size,                # int (logical vector length)
    BLOCK_SIZE: tl.constexpr,
    MAX_IDX: tl.constexpr,
):
    """
    Triton implementation of the element‑wise addition kernel.
    Mirrors the CUDA logic:
        if (global_idx < 2304 && global_idx < size)
            T_add[global_idx] = A[global_idx] + B[global_idx];
    """
    pid = tl.program_id(0)                # block index
    block_start = pid * BLOCK_SIZE         # first element processed by this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # global indices for this block

    # Valid indices are those less than both the logical size and the hard‑coded bound.
    mask = (offsets < size) & (offsets < MAX_IDX)

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(T_add_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original `cuda_kernel` API.
    Parameters
    ----------
    A, B, C : torch.Tensor
        Input and output tensors (float32) on the same CUDA device.
    size : int
        Logical vector length used to compute the launch grid.
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"

    # Ensure contiguous layout; for the output we require contiguity.
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    assert C.is_contiguous(), "Output tensor C must be contiguous"

    # Flatten to 1‑D views (no copy for C)
    A_flat = A.view(-1)
    B_flat = B.view(-1)
    C_flat = C.view(-1)

    # Verify that the flattened tensors have at least `size` elements.
    assert A_flat.numel() >= size and B_flat.numel() >= size and C_flat.numel() >= size, \
        "Tensor sizes must be at least `size`"

    # Compute grid dimensions exactly like the original CUDA host code
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A_flat,
        B_flat,
        C_flat,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_IDX=MAX_IDX,
        num_warps=32,   # 32 warps * 32 threads/warp = 1024 threads per program
    )