import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise addition A + B -> C
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    A_ptr, B_ptr, C_ptr: pointers to float32 tensors.
    size: total number of elements to process.
    BLOCK_SIZE: number of threads per block (compile-time constant).
    """
    # Program ID corresponds to block index in 1D grid
    pid = tl.program_id(0)
    # Compute absolute offsets for each thread
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask out-of-bounds threads
    mask = offsets < size
    # Load values (masked)
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    # Store the result
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point mirroring the original CUDA kernel signature.
    A, B, C must be CUDA tensors of dtype torch.float32.
    size specifies the number of elements to process.
    """
    # Basic validation
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Tensors must be of type torch.float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "Tensors must be contiguous"
    # Ensure size does not exceed tensor lengths
    max_len = min(A.numel(), B.numel(), C.numel())
    assert size <= max_len, f"size ({size}) exceeds tensor length ({max_len})"
    # Triton configuration
    BLOCK_SIZE = 1024  # matches __launch_bounds__(1024) in CUDA
    # Compute grid size (ceil division)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    # Launch kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK)

# Example usage (uncomment to run)
# __name__ == "__main__":
#     size = 2048000
#     A = torch.randn(size, device='cuda', dtype=torch.float32)
#     B = torch.randn(size, device='cuda', dtype=torch.float32)
#     C = torch.empty_like(A)
#     triton_kernel(A, B, C, size)
#     # Verify correctness
#     torch.testing.assert_allclose(C, A + B)