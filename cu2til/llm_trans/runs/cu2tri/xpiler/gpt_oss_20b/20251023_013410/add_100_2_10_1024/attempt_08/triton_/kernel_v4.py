import torch
import triton
import triton.language as tl

# Constants
BLOCK_SIZE = triton.language.constexpr(1024)
CHUNK = triton.language.constexpr(262144)
OUTER = triton.language.constexpr(8)

@triton.jit
def _iton_kernel_impl(A_ptr, B_ptr, C_ptr, size: tl.int32):
    # Offset for each thread within the block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    size64 = tl.cast(size, tl.int64)

    for outer in range(OUTER):
        idx = outer * tl.int64(CHUNK) + offset
        mask = idx < size64
        a = tl.load(A_ptr + idx, mask=mask)
        b = tl.load(B_ptr + idx, mask=mask)
        tl.store(C_ptr + idx, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    assert A.is_cuda and B.is_cuda and C.is_cuda, "Tensors must be on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "Tensors must be float32"
    assert A.numel() == size and B.numel() == and C.numel() == size, "Tensor sizes must match size"

    grid = (256,)  # 256 blocks, 1024 threads per block
    _triton_kernel_impl[grid](A, B, C, size)

if __name__ == "__main__":
    size = 2048000
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    triton_kernel(A, B, C, size)
    torch.testing.assert_allclose(C, A + B)
    print("Success")