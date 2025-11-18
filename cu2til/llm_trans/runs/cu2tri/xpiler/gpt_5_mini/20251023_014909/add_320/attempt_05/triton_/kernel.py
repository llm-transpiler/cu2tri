import torch
import triton
import triton.language as tl


# Triton kernel implementing elementwise addition.
# Must be named exactly as requested.
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK: tl.constexpr):
    """
    A_ptr, B_ptr, T_add_ptr: pointers to float32 arrays
    size: number of elements (int)
    BLOCK: number of elements processed per program (constexpr, must be power-of-two)
    """
    pid = tl.program_id(0)
    # create 0 .. BLOCK-1 (BLOCK must be a power of two for tl.arange)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Wrapper matching the CUDA kernel signature:
      triton_kernel(float *A, float *B, float *C, int size)

    A, B, C: 1-D torch.cuda.FloatTensor (or convertible)
    size: int number of elements to process
    """
    # Validate inputs
    if not torch.is_tensor(A) or not torch.is_tensor(B) or not torch.is_tensor(C):
        raise TypeError("A, B, C must be torch tensors")

    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("A, B, C must be torch.float32 tensors")

    if not A.is_cuda or not B.is_cuda or not C.is_cuda:
        raise ValueError("A, B, C must be CUDA tensors")

    size = int(size)
    if size < 0:
        raise ValueError("size must be non-negative")

    # Flatten and ensure contiguous
    A = A.contiguous().view(-1)
    B = B.contiguous().view(-1)
    C = C.contiguous().view(-1)

    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError("A, B, and C must have at least 'size' elements")

    # Triton requires tl.arange's range to be a power of two.
    # Choose a power-of-two BLOCK that provides good occupancy.
    # Using 256 as a practical value (power-of-two and multiple of 32).
    BLOCK = 256
    if size == 0:
        return

    num_blocks = (size + BLOCK - 1) // BLOCK
    grid = (num_blocks,)

    # Launch Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK=BLOCK)


# Optional self-test when run as a script.
if __name__ == "__main__":
    n = 1024 + 5  # test non-multiple length
    a = torch.randn(n, dtype=torch.float32, device='cuda')
    b = torch.randn(n, dtype=torch.float32, device='cuda')
    c = torch.empty(n, dtype=torch.float32, device='cuda')

    triton_kernel(a, b, c, n)
    torch.cuda.synchronize()
    assert torch.allclose(c, a + b)
    print("OK")