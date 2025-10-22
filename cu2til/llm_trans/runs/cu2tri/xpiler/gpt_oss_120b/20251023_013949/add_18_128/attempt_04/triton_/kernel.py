import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: element‑wise addition with the same guard as the CUDA version
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # float* __restrict__ A
    B_ptr,               # float* __restrict__ B
    C_ptr,               # float* __restrict__ T_add (output)
    BLOCK_SIZE: tl.constexpr = 1024,
    MAX_ELEMENTS: tl.constexpr = 2304,
):
    # program id = block index (1‑D grid)
    pid = tl.program_id(0)
    # Global linear index for each thread in the block
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Guard identical to the CUDA `if (global_idx < 2304)`
    mask = offset < MAX_ELEMENTS

    # Load with mask (out‑of‑range lanes get 0.0, which are never used)
    a = tl.load(A_ptr + offset, mask=mask, other=0.0)
    b = tl.load(B_ptr + offset, mask=mask, other=0.0)

    # Compute and store the result
    tl.store(C_ptr + offset, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper that mimics the original CUDA launch configuration
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point matching the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A, B, C : torch.cuda.FloatTensor
        and tensors ( be on the same CUDA device).
 size :
        Logical size of the vectors. Used only to compute the grid size,
        exactly like the original CUDA launch.
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"

    # Compute grid dimensions exactly as the CUDA code does:
    #   dim3 blockSize(1024);
    #   dim3 numBlocks((size + 1024 - 1) / 1024);
    grid = ((size + 1024 - 1) // 1024,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C)