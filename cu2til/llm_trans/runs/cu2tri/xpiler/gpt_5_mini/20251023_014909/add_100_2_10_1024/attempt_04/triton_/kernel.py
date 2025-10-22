import torch
import triton
import triton.language as tl

# Mirror the CUDA launch/config constants from the original code
_BLOCK = 1024            # threads per block
_NUM_BLOCKS = 256        # number of blocks
_STRIDE = _NUM_BLOCKS * _BLOCK  # 262144
_BOUND = 2048000         # hard-coded bound used in the original CUDA kernel

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_ptr,
                        BLOCK: tl.constexpr, STRIDE: tl.constexpr, BOUND: tl.constexpr):
    """
    Triton implementation of the original CUDA kernel:
      for outer in range(8):
        index = outer*STRIDE + blockIdx.x*BLOCK + threadIdx.x
        if index < BOUND:
          T[index] = A[index] + B[index]
    BLOCK, STRIDE and BOUND are passed as constexpr parameters so they are
    available at compile time inside the Triton kernel.
    """
    block_idx = tl.program_id(0)
    offs = block_idx * BLOCK + tl.arange(0, BLOCK)  # vector of per-program lanes

    # Unrolled outer loop (8 iterations) to match the original CUDA kernel
    for i in range(8):
        idx = offs + i * STRIDE
        mask = idx < BOUND
        a = tl.load(A_ptr + idx, mask=mask, other=0.0)
        b = tl.load(B_ptr + idx, mask=mask, other=0.0)
        tl.store(T_ptr + idx, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Wrapper matching the original CUDA launcher signature:
      extern "C" void cuda_kernel(float *A, float *B, float *C, int size)
    - A, B, C: torch.cuda.FloatTensor (device tensors)
    - size: present to match signature but ignored here to preserve original behavior
            (original CUDA launcher did not forward 'size' to the kernel; it used a
             hard-coded bound).
    This wrapper configures the grid (256 programs) and launches the Triton kernel
    with BLOCK=1024, STRIDE=262144 and BOUND=2048000 to preserve identical semantics.
    """
    # Basic type/device checks
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor on a CUDA device with dtype torch.float32")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise ValueError("All tensors must have dtype torch.float32")

    # Make contiguous copies if needed (and copy result back if original C was non-contiguous)
    A_orig, B_orig, C_orig = A, B, C
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    # Launch the Triton kernel with the same grid configuration as the CUDA launcher
    grid = (_NUM_BLOCKS,)
    _triton_kernel_impl[grid](A, B, C, BLOCK=_BLOCK, STRIDE=_STRIDE, BOUND=_BOUND)

    # If we created a separate contiguous output tensor, copy the results back to the original
    if C is not C_orig:
        C_orig.copy_(C)