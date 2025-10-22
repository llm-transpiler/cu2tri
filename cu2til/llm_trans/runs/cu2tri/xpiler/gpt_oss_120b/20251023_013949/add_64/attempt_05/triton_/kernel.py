import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise addition.
# The kernel name must be exactly `_triton_kernel_impl`.
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *Pointer* to first input tensor (float32)
    B_ptr,          # *Pointer* to second input tensor (float32)
    C_ptr,          # *Pointer* to output tensor (float32)
    size,           # Number of elements to process (int32)
    BLOCK_SIZE: tl.constexpr  # Compile‑time constant: threads per program
):
    # Program (block) identifier
    pid = tl.program_id(0)

    # Compute the linear indices this program will handle
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Guard against out‑of‑bounds accesses
    mask = offsets < size

    # Load, compute, and store
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mirrors the original CUDA kernel signature:
        cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32 on CUDA.
    size : int
        Number of elements to process.
    """
    # Basic sanity checks (optional but helpful)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32."
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor sizes must be at least `size`."

    # Define the block (program) size – matches the CUDA launch bounds of 64
    BLOCK_SIZE = 64

    # Compute grid dimensions: one program per BLOCK_SIZE elements
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE
    )