import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel equivalent to the original CUDA implementation:
    C[index] = A[index] + B[index] for index < size.
    Mirrors the launch configuration of 256 blocks × 1024 threads
    with an inner loop of 8 iterations.
    """
    pid = tl.program_id(0)                     # blockIdx.x
    base_offset = pid * BLOCK_SIZE             # blockIdx.x * 1024
    offsets = tl.arange(0, BLOCK_SIZE)         # threadIdx.x (int32)

    # Cast size to int32 for the comparison (optional but safe)
    size_i32 = tl.int32(size)

    # Outer loop corresponds to the fused loop in the CUDA code.
    for outer in range(8):
        # Compute the global index for this iteration.
        index = outer * 256 * BLOCK_SIZE + base_offset + offsets
        mask = index < size_i32

        a = tl.load(A_ptr + index, mask=mask, other=0.0)
        b = tl.load(B_ptr + index, mask=mask, other=0.0)
        tl.store(C_ptr + index, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that launches the Triton kernel.
    Parameters match the original CUDA kernel signature.
    """
    # Basic validation (mirrors expectations of original CUDA kernel)
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)
    assert A.is_cuda and B.is_cuda and C.is_cuda
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32
    assert A.numel() == size and B.numel() == size and C.numel() == size

    # Ensure contiguous layout for optimal performance
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    BLOCK_SIZE = 1024
    # Original launch configuration: 256 blocks, each with 1024 threads
    grid = (256,)

    # Launch the kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

    # Synchronize to guarantee completion before returning
    torch.synchronize()