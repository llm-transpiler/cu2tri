import torch
import triton
import triton.language as tl

# Triton kernel implementing the same logic as the original CUDA kernel.
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # float* __restrict__ A
    B_ptr,               # float* __restrict__ B
    T_add_ptr,           # float* __restrict__ T_add
    BLOCK_SIZE: tl.constexpr
):
    # Compute a linear index for each thread in the 1‑D grid.
    pid = tl.program_id(0)                     # blockIdx.x
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # threadIdx.x within block

    # The original CUDA kernel guards with a hard‑coded bound of 4032.
    mask = offsets < 4032

    # Load A and B, respecting the mask to avoid illegal memory accesses.
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    # Compute the element‑wise sum.
    t = a + b

    # Store the result back to T_add.
    tl.store(T_add_ptr + offsets, t, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original ``cuda_kernel`` entry point.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA device).
    B : torch.Tensor
        Input tensor B (float32, CUDA device).
    C : torch.Tensor
        Output tensor C (float32, CUDA device). Must have the same shape as A and B.
    size : int
        Logical size of the vectors. Used only to compute the launch grid.
    """
    # Basic sanity checks.
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.shape == B.shape == C.shape, "All tensors must have the same shape"

    # Ensure contiguous memory layout for pointer arithmetic.
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    BLOCK_SIZE = 1024  # Matches __launch_bounds__(1024) in the CUDA kernel.

    # Compute grid dimensions exactly as the original CUDA launch.
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel.
    _triton_kernel_impl[grid](
        A, B, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Optional: synchronize to make the kernel completion explicit.
    # torch.cuda.synchronize()