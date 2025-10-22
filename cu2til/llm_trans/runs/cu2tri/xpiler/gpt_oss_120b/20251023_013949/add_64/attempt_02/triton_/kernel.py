import torch
import triton
import triton.language as tl

# Triton kernel implementation
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper mimicking the original CUDA kernel signature:
    void cuda_kernel(float *A, float *B, float *C, int size)
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "Tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "Tensors must be contiguous"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, "Tensor sizes must be >= size"

    BLOCK_SIZE = 64
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

# Example usage
if __name__ == "__main__":
    size = 1024
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)
    triton_kernel(A, B, C, size)
    torch.testing.assert_allclose(C, A + B, atol=1e-6, rtol=1e-6)
    print("Triton kernel executed correctly.")