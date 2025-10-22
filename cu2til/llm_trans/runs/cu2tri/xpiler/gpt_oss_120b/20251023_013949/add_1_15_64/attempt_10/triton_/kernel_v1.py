import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that performs element‑wise addition:
    C[i] = A[i] + B[i] for i in [0, size).
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices within the block
    mask = offsets < size                       # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA kernel signature.
    Launches the Triton kernel with the same launch configuration
    (960 threads per block) and computes C = A + B element‑wise.
    """
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must reside on the CUDA device.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError("All tensors must be of type torch.float32.")

    BLOCK_SIZE = 960  # matches the original __launch_bounds__(960)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)  # number of blocks needed
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

# ----------------------------------------------------------------------
# Example usage (can be removed when integrating into a larger code base)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    size = 12345
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    triton_kernel(A, B, C, size)
    torch.cuda.synchronize()

    # Verify correctness
    assert torch.allclose(C, A + B), "Verification failed: results do not match."
    print("Triton kernel executed successfully.")