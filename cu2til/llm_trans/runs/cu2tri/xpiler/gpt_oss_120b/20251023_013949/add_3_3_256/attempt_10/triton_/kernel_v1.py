import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise addition with bounds checking.
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *Pointer* to first input tensor (float32)
    B_ptr,          # *Pointer* to second input tensor (float32)
    C_ptr,          # *Pointer* to output tensor (float32)
    size,           # Number of elements to process (int32)
    BLOCK_SIZE: tl.constexpr  # Compile‑time constant: threads per program
):
    # Compute a 1‑D index for each thread.
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask out‑of‑range threads.
    mask = offsets < size

    # Load, compute, and store.
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA device)
    B : torch.Tensor
        Input tensor (float32, CUDA device)
    C : torch.Tensor
        Output tensor (float32, CUDA device)
    size : int
        Number of elements to process
    """
    # Basic sanity checks – they mirror the expectations of the CUDA code.
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor storage must be at least `size` elements"

    BLOCK_SIZE = 1024  # Matches __launch_bounds__(1024) in the CUDA version

    # Compute grid dimensions (1‑D grid).
    grid = ( (size + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    # Launch the Triton kernel.
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )