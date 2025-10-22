import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A, B, C, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < size
    a = tl.load(A + pid, mask=mask, other=0.0)
    b = tl.load(B + pid, mask=mask, other=0.0)
    tl.store(C + pid, a + b, mask=mask)

def triton_kernel(A, B, C, size):
    BLOCK_SIZE = 1024  # power of 2 for triton arange
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    _triton_kernel_impl[grid, BLOCK_SIZE](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

if __name__ == "__main__":
    size = 10000
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty(size, device="cuda", dtype=torch.float32)
    triton_kernel(A, B, C, size)
    expected = A + B
    if torch.allclose(C, expected):
        print("Success: Triton kernel produced correct results.")
    else:
        print("Error: Results do not match.")