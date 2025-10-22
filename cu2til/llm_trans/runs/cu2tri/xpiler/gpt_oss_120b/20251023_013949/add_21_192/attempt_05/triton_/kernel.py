import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: elementwise addition A + B -> C
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024  # matches the CUDA block size

@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float
    B_ptr,          # *float
    C_ptr,          # *float
    size: tl.int32, # total number of elements to process
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton implementation of the CUDA kernel:
        if (idx < size) C[idx] = A[idx] + B[idx];
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices within block
    mask = offsets < size                       # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function: mimics the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that launches the Triton kernel.
    Parameters:
        A (torch.Tensor): input tensor of shape (size,) and dtype torch.float32
        B (torch.Tensor): input tensor of shape (size,) and dtype torch.float32
        C (torch.Tensor): output tensor of shape (size,) and dtype torch.float32
        size (int): number of elements to process (must be <= A.numel(), B.numel(), C.numel())
    """
    # Basic sanity checks (mirrors typical CUDA expectations)
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "Tensors must be contiguous"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "Tensors must be float32"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), "size exceeds tensor length"

    # Compute grid dimensions (one-dimensional grid)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# ----------------------------------------------------------------------
# Example usage (can be removed or commented out in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example: vector addition of length 4032 (as in the original CUDA example)
    N = 4032
    a = torch.randn(N, dtype=torch.float32, device="cuda")
    b = torch.randn(N, dtype=torch.float32, device="cuda")
    c = torch.empty_like(a)

    triton_kernel(a, b, c, N)

    # Verify correctness
    torch.testing.assert_allclose(c, a + b)
    print("Triton kernel executed successfully and results are correct.")