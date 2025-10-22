import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise addition
@triton.jit
def _triton_kernel_impl(A, B, C, size, BLOCK_SIZE: tl.constexpr):
    """
    Compute C[i] = A[i] + B[i] for i < size.
    """
    pid = tl.program_id(0)                     # program (block) index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global offsets
    mask = offsets < size                       # guard against out-of-bounds

    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, a + b, mask=mask)

def triton_kernel(A, B, C, size):
    """
    Triton entry point matching the original CUDA kernel signature.
    Parameters:
        A (torch.Tensor): Input tensor of shape (size,) and dtype torch.float32 on CUDA.
        B (torch.Tensor): Input tensor of shape (size,) and dtype torch.float32 on CUDA.
        C (torch.Tensor): Output tensor of shape (size,) and dtype torch.float32 on CUDA.
        size (int): Number of elements to process.
    """
    # Basic validation
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor), \
        "A, B, and C must be torch.Tensor objects"
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor sizes must be at least 'size'"

    BLOCK_SIZE = 1024  # matches the CUDA block size
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # one-dimensional grid
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

# Example usage / sanity check
if __name__ == "__main__":
    # Problem size (same as the CUDA example)
    N = 2304
    # Allocate input/output tensors on the GPU
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    # Launch the Triton kernel
    triton_kernel(A, B, C, N)

    # Verify correctness
    torch.testing.assert_allclose(C, A + B, atol=1e-6, rtol=1e-6)
    print("Triton kernel executed successfully and results are correct.")