import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that adds two vectors element‑wise.
    Mirrors the CUDA kernel:
        T_add[tid] = A[tid] + B[tid];
    """
    pid = tl.program_id(0)                     # program (block) index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global thread indices
    mask = offsets < N                          # guard against out‑of‑bounds
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that reproduces the original CUDA kernel signature:
        cuda_kernel(float *A, float *B, float *C, int size)
    All tensors must be torch.float32 on the same CUDA device.
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    BLOCK_SIZE = 64
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)  # one program per block of 64 elements
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

if __name__ == "__main__":
    # Simple sanity check
    size = 1024
    device = torch.device("cuda")
    A = torch.randn(size, device=device, dtype=torch.float32)
    B = torch.randn(size, device=device, dtype=torch.float32)
    C = torch.empty_like(A)
    triton_kernel(A, B, C, size)
    torch.testing.assert_allclose(C, A + B, atol=1e-6, rtol=1e-6)
    print("Triton kernel executed correctly.")