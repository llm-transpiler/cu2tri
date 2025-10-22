import torch
import triton
import triton.language as tl

# Compile‑time constants (mirroring the original CUDA kernel)
BLOCK_SIZE = 1024          # threads per block
MAX_ELEMENTS = 2304        # guard constant used in the CUDA kernel

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, BLOCK_SIZE: tl.constexpr):
    """Elementwise addition kernel (T_add = A + B) with the same guard as CUDA."""
    pid = tl.program_id(0)                     # block index
    offsets = tl.arange(0, BLOCK_SIZE)         # thread indices within the block
    idx = pid * BLOCK_SIZE + offsets            # global linear index
    mask = idx < MAX_ELEMENTS                   # replicate CUDA's < 2304 guard
    a = tl.load(A_ptr + idx, mask=mask, other=0.0)
    b = tl.load(B_ptr + idx, mask=mask, other=0.0)
    tl.store(C_ptr + idx, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int) -> None:
    """
    Triton entry point mirroring the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)
    """
    # Basic sanity checks (mirroring typical CUDA expectations)
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise RuntimeError("All tensors must be of type torch.float32")
    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    # Compute grid size exactly as the original CUDA launch configuration
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    # Launch the Triton kernel with the compile‑time BLOCK_SIZE argument
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE=BLOCK_SIZE)

# ----------------------------------------------------------------------
# Example usage / sanity test (can be removed in production)
if __name__ == "__main__":
    # Choose a size larger than the guard to demonstrate masking
    size = 5000
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)
    # Execute the kernel
    triton_kernel(A, B, C, size)
    # Verify correctness for the range the kernel actually processes (0..2303)
    expected = A[:MAX_ELEMENTS] + B[:MAX_ELEMENTS]
    if not torch.allclose(C[:MAX_ELEMENTS], expected, atol=1e-6):
        raise AssertionError(f"Result mismatch for the first {MAX_ELEMENTS} elements")
    print("Triton kernel executed successfully and passed verification.")