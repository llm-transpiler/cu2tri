import triton
import triton.language as tl
import torch
import math

# Triton kernel implementation (must be named exactly as requested)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK: tl.constexpr, VL: tl.constexpr):
    # VL must be a power-of-two. We create a vector of lane indices of length VL.
    offs = tl.arange(0, VL)               # VL is constexpr and a power-of-two
    mask = offs < BLOCK                   # only lanes < BLOCK are active (BLOCK may be non-power-of-two)

    # Load with mask (other=0.0 for inactive lanes)
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # Compute and store with mask
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Triton wrapper mirroring the CUDA host signature:
      triton_kernel(float *A, float *B, float *C, int size)

    Parameters:
      A, B, C : torch.Tensor (CUDA tensors of dtype torch.float32, contiguous is recommended)
      size     : int

    This wrapper configures the grid to match the original CUDA host code:
      blockSize = 960
      numBlocks = ceil(size / 960)
    and launches one Triton program per original CUDA block. The Triton kernel
    intentionally does NOT use the program id (to remain functionally equivalent
    to the provided CUDA kernel which used only threadIdx.x).
    """
    # Basic runtime checks
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor")
    if A.device.type != "cuda" or B.device.type != "cuda" or C.device.type != "cuda":
        raise ValueError("A, B, C must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("A, B, C must have dtype torch.float32")

    # Original CUDA block size
    BLOCK = 960
    # Compute VL = next power-of-two >= BLOCK (Triton's arange requires a power-of-two length)
    if BLOCK <= 0:
        return
    VL = 1 << ((BLOCK - 1).bit_length())  # e.g., BLOCK=960 -> VL=1024

    # Replicate CUDA's numBlocks = (size + 960 - 1) // 960
    numBlocks = (int(size) + BLOCK - 1) // BLOCK
    if numBlocks <= 0:
        # No blocks to launch (matching CUDA behavior)
        return

    # Ensure tensors are contiguous to provide predictable pointer arithmetic.
    # If they are not contiguous, make contiguous copies for A and B; for C we require contiguity.
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        # To preserve behavior similar to direct device pointer access in CUDA,
        # we require C to be contiguous. Copying C would mean writes go to a temporary.
        # It's safer to force contiguity by making C contiguous and then copying back.
        # However, the original CUDA kernel assumed contiguous flat pointers,
        # so here we make C contiguous and after kernel execution copy results back.
        C_orig = C
        C = C.contiguous()
        do_copy_back = True
    else:
        do_copy_back = False

    # Launch one Triton program per original CUDA block.
    grid = (numBlocks,)
    _triton_kernel_impl[grid](A, B, C, int(size), BLOCK=BLOCK, VL=VL)

    # If we created a contiguous temp for C, copy results back to the original view
    if do_copy_back:
        C_orig.copy_(C)