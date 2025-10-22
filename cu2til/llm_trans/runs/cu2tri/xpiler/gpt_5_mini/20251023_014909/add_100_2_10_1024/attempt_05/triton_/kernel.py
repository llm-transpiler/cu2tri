import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, SIZE, BLOCK: tl.constexpr):
    """
    Triton kernel implementing the same indexing as the original CUDA kernel:

    index = outer * (256 * 1024) + blockIdx.x * 1024 + threadIdx.x
    where:
      - blockIdx.x in [0, 256)
      - threadIdx.x in [0, 1024)
      - outer in [0, 8)

    Parameters:
      A_ptr, B_ptr, T_add_ptr: pointers to float32 device memory (passed as torch tensors)
      SIZE: number of valid elements (int scalar)
      BLOCK: number of elements processed per program (constexpr, 1024)
    """
    block_idx = tl.program_id(0)   # maps to CUDA's blockIdx.x (0..255)
    outer_idx = tl.program_id(1)   # maps to the outer loop (0..7)

    # NB = 256 (number of blocks along blockIdx.x in the original launch)
    start = outer_idx * (256 * BLOCK) + block_idx * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < SIZE

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Entry-point wrapper that mirrors the original host launcher signature:
      cuda_kernel(float *A, float *B, float *C, int size)

    Parameters:
      A, B, C: torch.cuda.FloatTensor (any shape) -- will be viewed as 1D
      size: int -- number of elements to process (equivalent to the bound check in CUDA)
    """
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor on CUDA")

    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors")

    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must be torch.float32")

    # Flatten to 1-D views (no copies)
    A_ = A.view(-1)
    B_ = B.view(-1)
    C_ = C.view(-1)

    if size > A_.numel() or size > B_.numel() or size > C_.numel():
        raise ValueError("size is larger than the provided tensors' number of elements")

    # Match the original CUDA launch configuration:
    # blockSize = 1024, numBlocks = 256, outer loop range = 8
    num_blocks = 256
    block_size = 1024
    outer = 8
    grid = (num_blocks, outer)

    # Launch Triton kernel. BLOCK is a compile-time constant here.
    _triton_kernel_impl[grid](A_, B_, C_, size, BLOCK=block_size)