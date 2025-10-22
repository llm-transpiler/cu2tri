import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    idx = offset + tl.arange(0, BLOCK_SIZE)
    mask = idx < size
    a = tl.load(A_ptr + idx, mask=mask, other=0.0)
    b = tl.load(B_ptr + idx, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + idx, c, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton wrapper that mimics the original CUDA kernel interface.
    Parameters:
        A (torch.Tensor): Input tensor of shape (size,) and dtype torch.float32.
        B (torch.Tensor): Input tensor of shape (size,) and dtype torch.float32.
        C (torch.Tensor): Output tensor of shape (size,) and dtype torch.float32.
        size (int): Number of elements to process.
    """
    # Validate inputs
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "Tensors must be contiguous"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "Tensors must be float32"
    assert size <= A.numel() and size <= B.numel() and size <= C.numel(), "size exceeds tensor length"
    BLOCK_SIZE = 1024
    # Compute grid as a tuple (required by Triton)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    # Launch kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()

# Example usage (can be removed if not needed)
if __name__ == "__main__":
    size = 4096
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)
    triton_kernel(A, B, C, size)
    torch.testing.assert_allclose(C, A + B, atol=1e-6, rtol=1e-6)
    print("Kernel executed successfully.")