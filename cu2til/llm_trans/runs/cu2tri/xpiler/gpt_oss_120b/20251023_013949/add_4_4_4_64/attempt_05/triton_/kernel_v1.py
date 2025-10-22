import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise addition with a hardcoded bound of 4096 elements.
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # *Pointer* to the first input array (float32)
    B_ptr,               # *Pointer* to the second input array (float32)
    T_add_ptr,           # *Pointer* to the output array (float32)
    BLOCK_SIZE: tl.constexpr  # Compile‑time constant: number of threads per program
):
    # Compute a linear index for each thread.
    pid = tl.program_id(0)                     # Program (block) index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Guard against out‑of‑range accesses – exactly the same condition as the CUDA kernel.
    mask = offsets < 4096

    # Load values from A and B, using 0.0 for masked‑off lanes.
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    # Compute the elementwise sum.
    c = a + b

    # Write the result back to T_add.
    tl.store(T_add_ptr + offsets, c, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original CUDA launch interface:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor 1 (float32, contiguous).
    B : torch.Tensor
        Input tensor 2 (float32, contiguous).
    C : torch.Tensor
        Output tensor (float32, contiguous). Must have at least `size` elements.
    size : int
        Logical size of the vectors (used only for grid sizing, identical to the CUDA wrapper).
    """
    # Basic sanity checks – these mirror typical expectations of the CUDA code.
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "All tensors must be contiguous"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "All tensors must be float32"
    # The original CUDA kernel uses a fixed block size of 1024 threads.
    BLOCK_SIZE = 1024

    # Compute the grid dimension exactly as the CUDA host code does.
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel. The BLOCK_SIZE argument is passed as a compile‑time constant.
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE=BLOCK_SIZE)