import torch
import triton
import triton.language as tl

# Compile‑time block size matching the original CUDA launch bounds
BLOCK_SIZE = 320

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton implementation of:
        C[i] = A[i] + B[i]   for i in [0, size)
    One program (thread) processes BLOCK_SIZE contiguous elements.
    """
    pid = tl.program_id(0)                     # program (block) index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)   # indices this program works on

    # Guard against out‑of‑bounds when size is not a multiple of BLOCK_SIZE
    mask = offsets < size

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mirrors the original CUDA kernel signature:
        triton_kernel(float* A, float* B, float* C, int size)

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 CUDA tensors. They must be at least `size` elements long.
    size : int
        Number of elements to process.
    """
    # Basic sanity checks – they are cheap and help catch misuse early.
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), \
        "size exceeds tensor length"

    # Compute grid dimensions (one program per BLOCK_SIZE elements)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)


# Optional simple test when run as a script
if __name__ == "__main__":
    torch.manual_seed(0)
    N = 1024  # any size; not necessarily a multiple of BLOCK_SIZE
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    triton_kernel(a, b, c, N)

    # Verify correctness against PyTorch
    torch.testing.assert_allclose(c, a + b)
    print("Triton kernel produced correct results.")