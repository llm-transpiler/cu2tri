import torch
import triton
import triton.language as tl


# Triton kernel implementing elementwise addition.
# Named exactly as requested.
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK: tl.constexpr):
    # program (block) id
    pid = tl.program_id(0)
    # thread indices inside a block: 0 .. BLOCK-1
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # mask to avoid out-of-bounds accesses
    mask = offs < size
    # load, compute, store with mask
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Triton wrapper matching the CUDA kernel signature:
      triton_kernel(float *A, float *B, float *C, int size)

    Parameters:
    - A, B, C: 1-D torch.cuda.FloatTensor (or convertible). These are treated
               as flat pointers to float (like float* in CUDA).
    - size: int number of elements to process.

    The wrapper configures the grid and launches the Triton kernel.
    """
    # Input validation and normalization
    if not torch.is_tensor(A) or not torch.is_tensor(B) or not torch.is_tensor(C):
        raise TypeError("A, B, C must be torch tensors (CUDA tensors are required).")

    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("A, B, C must be torch.float32 tensors.")

    if not A.is_cuda or not B.is_cuda or not C.is_cuda:
        raise ValueError("A, B, C must be CUDA tensors (device='cuda').")

    # Flatten to 1-D pointer-like tensors and ensure contiguous memory
    A = A.contiguous().view(-1)
    B = B.contiguous().view(-1)
    C = C.contiguous().view(-1)

    size = int(size)
    if size < 0:
        raise ValueError("size must be non-negative")

    # Sanity check: ensure buffers are large enough for requested size
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError("A, B, and C must have at least 'size' elements.")

    # Block size chosen to mirror the original CUDA kernel's blockDim.x
    BLOCK = 320
    num_blocks = (size + BLOCK - 1) // BLOCK
    grid = (num_blocks,)

    # Launch Triton kernel. This is asynchronous like CUDA kernel launches by default.
    _triton_kernel_impl[grid](A, B, C, size, BLOCK=BLOCK)


# Optional test when run as a script (kept inside module; no external text output).
if __name__ == "__main__":
    # simple sanity check
    n = 1024
    a = torch.randn(n, dtype=torch.float32, device='cuda')
    b = torch.randn(n, dtype=torch.float32, device='cuda')
    c = torch.empty(n, dtype=torch.float32, device='cuda')

    triton_kernel(a, b, c, n)

    # verify correctness (this will raise if something is wrong)
    torch.cuda.synchronize()
    assert torch.allclose(c, a + b), "Result mismatch!"
    print("OK")