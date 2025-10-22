import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, B, C, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(A + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)
    tl.store(C + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    BLOCK_SIZE = 960
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

if __name__ == "__main__":
    size = 10000
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)
    triton_kernel(A, B, C, size)
    torch.testing.assert_allclose(C, A + B)
    print("Success!")