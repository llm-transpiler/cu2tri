import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise addition.
# Named exactly as required.
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Compute C[i] = A[i] + B[i] for i in [0, size).
    The kernel is launched with a 1‑D grid where each program processes
    BLOCK_SIZE consecutive elements.
    """
    # Program (block) index.
    pid = tl.program_id(0)
    # Offsets for this program.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask out-of‑range indices.
    mask = offsets < size
    # Load with mask (out‑of‑range loads return 0.0, but they are never used).
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    # Store the result.
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA kernel launch signature:
        triton_kernel(float* A, float* B, float* C, int size)

    Parameters
       A, B, C : torch.Tensor
        1‑D tensors of dtype torch.float32 residing on the same CUDA device.
    size : int
        Number of elements to process (must be <= A.numel(), B.numel(), C.numel()).
    """
    # Basic sanity checks – they are cheap and help catch misuse early.
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.ndim == 1 and B.ndim == 1 and C.ndim == 1, "Only 1‑D tensors are supported"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), \
        "size exceeds tensor length"

    # Triton block size – matches the CUDA launch bounds of 1024 threads.
    BLOCK_SIZE = 1024

    # Compute the 1‑D grid size needed to cover `size` elements.
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the kernel. Triton automatically extracts raw pointers from the tensors.
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

# Example usage (uncomment to run):
# if __name__ == "__main__":
#     N = 4096
#     a = torch.randn(N, device="cuda", dtype=torch.float32)
#     b = torch.randn(N, device="cuda", dtype=torch.float32)
#     c = torch.empty_like(a)
#     triton_kernel(a, b, c, N)
#     torch.testing.assert_allclose(c, a + b)