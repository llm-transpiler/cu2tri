import torch
import triton
import triton.language as tl

# Number of threads per program (matches CUDA block size)
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes:
        C[i] = A[i] + B[i]  for i in [0, size)
    Mirrors the behavior of the original CUDA kernel.
    """
    pid = tl.program_id(0)                     # program (block) index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices
    mask = offsets < size                       # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point wrapper that launches the Triton kernel.
    Parameters:
        A (torch.Tensor): Input tensor of shape (>=size,) and dtype float32.
        B (torch.Tensor): Input tensor of shape (>=size,) and dtype float32.
        C (torch.Tensor): Output tensor of shape (>=size,) and dtype float32.
        size (int): Number of elements to process.
    """
    # Validate device placement
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must reside on the CUDA device.")
    # Ensure contiguity
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()
    # Type checks
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("All tensors must be of type torch.float32.")
    # Size sanity
    if size < 0:
        raise ValueError("size must be non‑negative.")
    if size == 0:
        return

    # Compute grid dimensions (1‑D grid)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the kernel with the appropriate compile‑time and launch parameters
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32,  # 32 threads per warp → 1024 threads per block
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()


if __name__ == "__main__":
    # Simple correctness test (mirrors the original CUDA usage)
    N = 2304  # matches the hard‑coded bound in the CUDA kernel
    A = torch.randn(N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, dtype=torch.float32, device="cuda")
    C = torch.empty_like(A)

    triton_kernel(A, B, C, N)

    # Verify results against PyTorch reference
    torch.testing.assert_allclose(C, A + B, atol=1e-6, rtol=1e-6)
    print("Triton kernel produced correct results.")