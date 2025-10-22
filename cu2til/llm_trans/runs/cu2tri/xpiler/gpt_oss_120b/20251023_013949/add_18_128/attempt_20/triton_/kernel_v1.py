import torch
import triton
import triton.language as tl

# Compile‑time constants (mirroring the CUDA launch bounds)
BLOCK_SIZE = 1024          # threads per block
MAX_ELEMENTS = 2304        # hard‑coded bound used in the original CUDA kernel

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr,
                        BLOCK_SIZE: tl.constexpr,
                        MAX_ELEMENTS: tl.constexpr):
    """
    Triton implementation of the element‑wise addition kernel.
    Mirrors the behavior of the original CUDA kernel:
        if (global_idx < 2304) {
            T_add[global_idx] = A[global_idx] + B[global_idx];
        }
    """
    pid = tl.program_id(0)                     # block index
    block_start = pid * BLOCK_SIZE              # first element handled by this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)   # indices for the threads in the block

    # Apply the same hard‑coded bound check as the CUDA version
    mask = offsets < MAX_ELEMENTS

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(T_add_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor,
                  B: torch.Tensor,
                  C: torch.Tensor,
                  size: int):
    """
    Entry‑point that mimics the original ``cuda_kernel`` signature.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA, contiguous).
    B : torch.Tensor
        Input tensor B (float32, CUDA, contiguous).
    C : torch.Tensor
        Output tensor C (float32, CUDA, contiguous).
    size : int
        Number of elements (used only for grid configuration, identical to the CUDA wrapper).
    """
    # Basic sanity checks – these mirror typical expectations of the CUDA launch code
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    # Ensure contiguous layout for pointer arithmetic inside the kernel
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # Compute the grid size exactly as the CUDA wrapper does:
    #   numBlocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (size + BLOCK - ) // BLOCK_SIZE

    # Launch the Triton kernel.  BLOCK_SIZE and MAX_ELEMENTS are passed as compile‑time arguments.
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE, MAX_ELEMENTS)