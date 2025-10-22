import torch
import triton
import triton.language as tl

# Constants
BLOCK_SIZE = triton.language.constexpr(1024)
OUTER = triton.language.constexpr(8)
CHUNK = triton.language.constexpr(262144)

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size: tl.int32):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    outer_offsets = tl.arange(0, OUTER) * CHUNK
    indices = outer_offsets + offset
    mask = indices < tl.cast(size, tl.int64)
    a = tl.load(A_ptr + indices, mask=mask    b = tl.load(B_ptr indices, mask=mask)
    tl.store(C_ptr + indices, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    assert A.is_cuda and B.is_cuda and C.is_cuda, "Tensors must be on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "Tensors must be float32"
    assert A.numel() == size and B.numel() == size and C.numel() == size, "Tensor sizes must match size"
    grid = (256,)
    _triton_kernel_impl[grid](A, B, C, size)

if __name__ == "__main__":
    size = 2048000
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)
    triton_kernel(A, B, C, size)
    torch.testing.assert_allclose(C, A + B)
    print("Success")