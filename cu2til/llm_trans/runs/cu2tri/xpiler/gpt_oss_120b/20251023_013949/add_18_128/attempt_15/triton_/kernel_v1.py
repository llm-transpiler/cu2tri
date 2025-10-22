import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Compile‑time constants (mirroring the original CUDA launch bounds)
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024          # __launch_bounds__(1024)
MAX_ELEMENTS = 2304        # hard‑coded guard in the CUDA kernel

@triton.jit
def _triton_kernel_impl(A, B, T_add,
                        BLOCK_SIZE: tl.constexpr,
                        MAX_ELEMENTS: tl.constexpr    """
    Triton implementation of the original CUDA.
    Performs elementwise + B → T_add for indices < 2304.
    The logic (including the guard) is identical to the CUDA version.
    """
    pid = tl.program_id(0)                     # block index (gridDim.x)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < MAX_ELEMENTS               # replicate CUDA's if‑condition

    a tl.load + offsets, mask=mask)        # load A[i] when i < 2304
    b = tl.load(B + offsets, mask=mask)        # load B[i] when i < 2304
    tl.store(T_add + offsets, a + b, mask=mask)  # write result

def triton_kernel(A: torch.Tensor,
                  B: torch.Tensor,
                  C: torch.Tensor,
                  size: int):
    """
    Entry point that matches the original `cuda_kernel` signature.
    Parameters
    ----------
    A, B, C : torch.Tensor
        CUDA tensors of dtype torch.float32.
    size : int
        Logical size of the vectors (used only for grid calculation,
        exactly as in the CUDA launch configuration).
    """
    # ------------------------------------------------------------------
    # Sanity checks – they correspond to the expectations of the CUDA code
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"

    # Compute grid dimensions exactly like the CUDA launch:
    #   dim3 blockSize(1024);
    #   dim3 numBlocks((size + 1024 - 1) / 1024);
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    # Launch the Triton kernel. Compile‑time constants are passed as keyword args.
    _triton_kernel_impl[grid](A, B, C,
                              BLOCK_SIZE=BLOCK_SIZE,
                              MAX_ELEMENTS=MAX_ELEMENTS)

    # Optional synchronization (useful for debugging / timing)
    # torch.cuda.synchronize()