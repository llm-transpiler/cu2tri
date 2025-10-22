import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < N
    A = tl.load(A_ptr + indices, mask=mask)
    B = tl.load(B_ptr + indices, mask=mask)
    C = A + B
    tl.store(C_ptr + indices, C, mask=mask)

def triton_kernel(A, B, C, size):
    BLOCK_SIZE = 1024
    N = 2048000
    grid = triton.cdiv(N, BLOCK_SIZE)
    _triton_kernel_impl[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)

if __name__ == "__main__":
    size = 2048000
    A = torch.randn(size, device='cuda', dtype=torch.float32)
    B = torch.randn(size, device='cuda', dtype=torch.float32)
    C = torch.empty_like(A)
    triton_kernel(A, B, C, size)
    expected = A + B
    if torch.allclose(C, expected, atol=1e-5, rtol=1e-5):
        print("Success!")
    else:
        print("Mismatch!")