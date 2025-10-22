import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes the sign of each element in A and writes it to C.
    Mirrors the behavior of the original CUDA kernel:
        - Uses a fixed bound of 4608 elements.
        - Returns 1.0 for positive values, -1.0 for negative values, 0.0 for zero.
    """
    pid = tl.program_id(0)  # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices
    # Only process indices < 4608 (same bound as the CUDA kernel)
    mask = offsets < 4608

    # Load A safely; out‑of‑bounds loads are masked to 0.0 (won't be used)
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # Compute sign: 1.0 if a > 0, -1.0 if a < 0, 0.0 otherwise
    sign = tl.sign(a)  # tl.sign returns -1, 0, or 1 for each element

    # Store the result back to C
    tl.store(C_ptr + offsets, sign, mask=mask)

def triton_kernel(A: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that launches the Triton kernel.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of the same shape and dtype as A.
    size : int
        Logical size of the vectors (mirrors the original CUDA API).
    """
    # Basic sanity checks
    assert A.is_cuda and C.is_cuda, "A and C must be CUDA tensors"
    assert A.dtype == torch.float32 and C.dtype == torch.float32, "Tensors must be float32"
    assert A.numel() >= size and C.numel() >= size, "Tensor storage must be at least 'size' elements"

    BLOCK_SIZE = 1024
    # Compute the number of blocks needed (same as CUDA launch configuration)
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the kernel
    _triton_kernel_impl[grid](A, C, size, BLOCK_SIZE=BLOCK_SIZE)

# Example usage (uncomment to test):
# if __name__ == "__main__":
#     size = 4608
#     A = torch.randn(size, device="cuda", dtype=torch.float32)
#     C = torch.empty_like(A)
#     triton_kernel(A, C, size)
#     # Verify correctness
#     expected = torch.sign(A)
#     assert torch.allclose(C, expected), "Result mismatch!"