import triton
import triton.language as tl
import torch

# Triton kernel implementation (must be named exactly as requested)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK: tl.constexpr):
    # Emulate the original CUDA kernel behavior:
    # Each CUDA block has 960 threads and each thread uses only threadIdx.x (0..959).
    # To replicate that exactly and efficiently in Triton, we map one Triton program
    # to one original CUDA block and operate on the range [0, BLOCK).
    offs = tl.arange(0, BLOCK)                 # represents threadIdx.x for threads 0..BLOCK-1
    # Note: original CUDA kernel did not account for blockIdx.x when indexing,
    # so every block writes the same indices [0..BLOCK-1]. We reproduce that behavior.
    a = tl.load(A_ptr + offs)
    b = tl.load(B_ptr + offs)
    tl.store(T_add_ptr + offs, a + b)


def triton_kernel(A, B, C, size):
    """
    Triton wrapper that mirrors the CUDA host function signature:
      triton_kernel(float *A, float *B, float *C, int size)

    Parameters:
      A, B, C : torch.Tensor (must be CUDA tensors of dtype torch.float32 and contiguous)
      size     : int

    Behavior:
      - Configures a grid with block size 960, number of blocks = ceil(size / 960)
      - Launches _triton_kernel_impl to reproduce the original CUDA kernel semantics
        (note: the original CUDA kernel used only threadIdx.x and did not use blockIdx.x).
    """
    # Basic runtime checks to ensure compatibility with Triton kernel assumptions
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor")
    if A.device.type != "cuda" or B.device.type != "cuda" or C.device.type != "cuda":
        raise ValueError("A, B, C must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("A, B, C must have dtype torch.float32")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise ValueError("A, B, C must be contiguous tensors")

    BLOCK = 960
    # Replicate CUDA's numBlocks = (size + 960 - 1) / 960
    numBlocks = (int(size) + BLOCK - 1) // BLOCK
    if numBlocks <= 0:
        # Matching CUDA behavior: if there are no blocks to launch, do nothing.
        return

    # Launch one Triton program per original CUDA block.
    # Each Triton program will process the vector of indices [0..BLOCK-1], matching threadIdx.x-only indexing.
    grid = (numBlocks,)
    # Note: pass BLOCK as a constexpr so Triton can optimize loads/stores
    _triton_kernel_impl[grid](A, B, C, int(size), BLOCK=BLOCK)