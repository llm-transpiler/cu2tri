import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise addition.
# Named exactly as required.
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # float* __restrict__ A
    B_ptr,          # float* __restrict__ B
    C_ptr,          # float* __restrict__ T_add (output)
    N,              # total number of elements to process
    BLOCK_SIZE: tl.constexpr  # compile‑time constant, matches CUDA launch bound
):
    # Program (block) identifier
    pid = tl.program_id(0)
    # Offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask out‑of‑range elements
    mask = offsets < N
    # Load, compute, and store
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mirrors the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA).
    B : torch.Tensor
        Input tensor B (float32, CUDA).
    C : torch.Tensor
        Output tensor C (float32, CUDA). Must have at least `size` elements.
    size : int
        Number of elements to process.
    """
    # Basic sanity checks – keep behavior identical to the CUDA version.
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor storage must be at least `size` elements"

    BLOCK_SIZE = 320  # matches __launch_bounds__(320) in the CUDA code
    # Compute grid size exactly as in the CUDA host code.
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel.
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)