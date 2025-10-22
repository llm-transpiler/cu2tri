import torch
import triton
import triton.language as tl

# Constants matching the original CUDA kernel launch/configuration
_BLOCK = 1024            # blockSize in CUDA
_NUM_BLOCKS = 256        # numBlocks in CUDA
_OUTER_ITERS = 8         # loop count in the CUDA kernel
_STRIDE = _NUM_BLOCKS * _BLOCK  # 262144
_BOUND = 2048000         # hard-coded bound used in the original CUDA kernel

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_ptr, BLOCK: tl.constexpr):
    """
    Triton kernel implementing the same elementwise addition performed by the
    CUDA kernel. Each Triton program corresponds to one CUDA block (blockIdx.x).
    Inside the program we iterate the same 8 outer iterations as the CUDA kernel,
    accessing indices: idx = outer * STRIDE + blockIdx.x * BLOCK + threadIdx.x
    with a vector of threadIdx (0..BLOCK-1) implemented via tl.arange.
    """
    block_idx = tl.program_id(0)                          # corresponds to blockIdx.x
    offs = block_idx * BLOCK + tl.arange(0, BLOCK)        # vector of threadIdx positions

    # Repeat the same outer-loop pattern as in CUDA kernel
    # and perform masked loads/stores to avoid out-of-bounds accesses.
    for i in range(8):
        idx = offs + i * (_STRIDE)                        # match idx = outer*262144 + base
        mask = idx < _BOUND
        a = tl.load(A_ptr + idx, mask=mask, other=0.0)
        b = tl.load(B_ptr + idx, mask=mask, other=0.0)
        tl.store(T_ptr + idx, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Wrapper function that mirrors the original CUDA launcher:
      extern "C" void cuda_kernel(float *A, float *B, float *C, int size)
    - Signature is identical (A, B, C, size). The original CUDA launcher ignored
      the 'size' argument and used hard-coded bounds inside the kernel; to keep
      behavior identical we do the same here.
    - This function configures the Triton grid to numBlocks=256 and launches
      the Triton kernel with BLOCK=1024.
    - A, B, C must be CUDA tensors of dtype torch.float32.
    """
    # Basic validation to keep types consistent with original float* semantics
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor on a CUDA device with dtype torch.float32")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be on CUDA device")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise ValueError("All tensors must have dtype torch.float32")
    # Require contiguous tensors to have stable device pointers (matching low-level CUDA pointer usage)
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise ValueError("A, B, C must be contiguous tensors")

    # Launch Triton kernel with the same grid configuration as the CUDA launcher
    grid = (_NUM_BLOCKS,)
    # NOTE: The original CUDA wrapper did not forward 'size' to the kernel; the kernel used a hard-coded bound.
    # To preserve identical behavior we do not use 'size' here and rely on the _BOUND constant inside the Triton kernel.
    _triton_kernel_impl[grid](A, B, C, BLOCK=_BLOCK)