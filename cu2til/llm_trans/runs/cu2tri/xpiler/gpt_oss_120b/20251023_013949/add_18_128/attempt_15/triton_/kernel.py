import torch
import triton
import triton.language as tl

# Compile‑time constants mirroring the original CUDA launch configuration
BLOCK_SIZE = 1024          # __launch_bounds__(1024)
MAX_ELEMENTS = 2304        # hard‑coded guard in the CUDA kernel

@triton.jit
def _triton_kernel_impl(A, B, T_add, size,
                        BLOCK_SIZE: tl.constexpr,
                        MAX_ELEMENTS: tl.constexpr):
    """
    Triton implementation of the original CUDA kernel.
    Performs element‑wise addition A + B → T_add for indices that satisfy
    both the hard‑coded limit (2304) and the actual logical size.
    """
    pid = tl.program_id(0)                     # block index (gridDim.x)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Combine the original guard with the runtime size to avoid OOB accesses
    mask = (offsets < MAX_ELEMENTS) & (offsets < size)

    a = tl.load(A + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)
    tl.store(T_add + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor,
                  B: torch.Tensor,
                  C: torch.Tensor,
                  size: int):
    """
    Entry point matching the original `cuda_kernel` signature.
    Parameters
    ----------
    A, B, C : torch.Tensor
        CUDA tensors of dtype torch.float32.
    size : int
        Logical size of the vectors (used for grid calculation and bounds).
    """
    # ------------------------------------------------------------------
    # Input validation – mirrors expectations of the original CUDA code
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert (A.dtype == torch.float32 and
            B.dtype == torch.float32 and
            C.dtype == torch.float32), "All tensors must be float32"

    # Compute grid dimensions exactly as in the CUDA launch:
    #   dim3 blockSize(1024);
    #   dim3 numBlocks((size + 1024 - 1) / 1024);
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel. Compile‑time constants are passed as keyword args.
    _triton_kernel_impl[grid](A, B, C, size,
                              BLOCK_SIZE=BLOCK_SIZE,
                              MAX_ELEMENTS=MAX_ELEMENTS)

    # Ensure kernel completion before returning (useful for testing / timing)
    torch.cuda.synchronize()