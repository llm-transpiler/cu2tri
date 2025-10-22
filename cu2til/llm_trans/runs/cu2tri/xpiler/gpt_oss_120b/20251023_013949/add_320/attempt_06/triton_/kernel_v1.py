import torch
import triton
import triton.language as tl

# Compile‑time block size (matches the CUDA bounds)
BLOCK_SIZE = 320

@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float
    B_ptr,          # *float
    C_ptr,          # *float
    size,           # i32 (unused for indexing, kept for signature compatibility)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that mirrors the original CUDA kernel:
    each thread adds A[threadIdx.x] + B[threadIdx.x] and writes the result to
    C[threadIdx.x].  The kernel does NOT use blockIdx.x, reproducing the exact
    behavior of the CUDA implementation.
    """
    # Thread indices within the block
    offs = tl.arange(0, BLOCK_SIZE)

    # Guard against out‑of‑bounds when size < BLOCK_SIZE
    mask = offs < size

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offs, c, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel with the same signature as the
    original CUDA entry point.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32 on CUDA.
    size : int
        Number of elements (mirrors the CUDA `size` argument).
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"

    # Grid: one block per BLOCK_SIZE chunk (identical to the CUDA launch configuration)
    grid = lambda meta: ((size + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    # Launch the kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Ensure completion before returning to host code
    torch.cuda.synchronize()


# Example usage (can be removed or adapted as needed)
if __name__ == "__main__":
    size = 1024
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    triton_kernel(A, B, C, size)

    # The original CUDA kernel only writes the first BLOCK_SIZE elements.
    # Verify those elements match the expected result.
    torch.testing.assert_close(C[:BLOCK_SIZE], A[:BLOCK_SIZE] + B[:BLOCK_SIZE])
    print("Triton kernel executed successfully.")