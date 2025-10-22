import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size,
                        BLOCK_SIZE: tl.constexpr = 1024,
                        MAX_ELEM: tl.constexpr = 2304):
    """Element‑wise addition with the same bounds as the original CUDA kernel."""
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices within block
    mask = (offsets < size) & (offsets < MAX_ELEM)         # enforce both size and 2304 limit

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton wrapper mimicking the original CUDA kernel signature.
    Parameters:
        A (torch.Tensor): 1‑D float32 CUDA tensor.
        B (torch.Tensor): 1‑D float32 CUDA tensor.
        C (torch.Tensor): 1‑D float32 CUDA tensor (output).
        size (int): Logical number of elements to process.
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous()
    BLOCK_SIZE = 1024
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # one‑dimensional grid
    _triton_kernel_impl[grid](A, B, C, size,
                              BLOCK_SIZE=BLOCK_SIZE,
                              MAX_ELEM=2304)

# Example usage (can be removed in production)
if __name__ == "__main__":
    # Choose a size ≤ 2304 to fully exercise the kernel
    size = 2000
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)
    triton_kernel(A, B, C, size)
    torch.cuda.synchronize()
    torch.testing.assert_allclose(C[:size], A[:size] + B[:size])
    print("Triton kernel executed correctly.")