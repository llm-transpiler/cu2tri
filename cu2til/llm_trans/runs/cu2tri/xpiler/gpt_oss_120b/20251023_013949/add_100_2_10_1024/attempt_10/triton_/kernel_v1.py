import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: element‑wise addition (direct translation of the CUDA kernel)
# ----------------------------------------------------------------------
@triton.jit
deftriton_kernel_impl(A_ptr, B_ptr, C_ptr, size,
                        BLOCK_SIZE: tl.constexpr = 1024):
    """
    Triton implementation that mirrors the original CUDA kernel.
    Each program (block) processes BLOCK_SIZE contiguous elements.
    """
    # Program (block) identifier
    pid = tl.program_id(0)

    # Offsets for the threads within this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Guard against out‑of‑bounds accesses
    mask = offsets < size

    # Load, compute, and store with masking
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int) -> None:
    """
    Entry‑point that mimics the original CUDA wrapper:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors allocated on the same CUDA device.
    size : int
        Number of elements to process. The original CUDA kernel ignored this
        argument and used a hard‑coded bound (2 048 000). Here we honour the
        passed value for flexibility while preserving identical behaviour
        when `size == 2048000`.
    """
    # Basic sanity checks – keep the behaviour close to the CUDA version
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor storage must be at least `size` elements"

    # Compute grid size: one program per BLOCK_SIZE elements
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(size, BLOCK_SIZE),)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)