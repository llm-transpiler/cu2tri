import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A, B, C, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, a + b, mask=mask)

def triton_kernel(A, B, C, size):
    BLOCK_SIZE = 1024
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    _triton_kernel_impl[(grid,)](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

if __name__ == "__main__":
    size = 10000
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty(size, device="cuda", dtype=torch.float32)
    triton_kernel(A, B, C, size)
    torch.testing.assert_allclose(C, A + B)
    print("Success!")