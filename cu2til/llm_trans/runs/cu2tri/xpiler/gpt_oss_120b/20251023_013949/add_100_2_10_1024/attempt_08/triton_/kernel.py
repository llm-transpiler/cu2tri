import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, B, C, BLOCK_SIZE: tl.constexpr):
    pid0 = tl.program_id(0)  # block index within a chunk
    pid1 = tl.program_id(1)  # outer chunk index (0..7)
    # Global offsets for each thread in the block
    offsets = pid1 * 262144 + pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Hard‑coded bound from the original CUDA kernel
    mask = offsets < 2048000
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton implementation of the element‑wise addition kernel.
    The signature mirrors the original CUDA kernel (including the unused `size` argument).
    The kernel processes exactly 2,048,000 elements, matching the original behavior.
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "All tensors must be contiguous"

    # Grid configuration: 256 blocks per outer chunk, 8 outer chunks (matches original launch)
    BLOCK_SIZE = 1024
    num_blocks = 256
    outer_chunks = 8
    grid = (num_blocks, outer_chunks)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )

if __name__ == "__main__":
    # Simple correctness test
    N = 2048000
    A = torch.randn(N, dtype=torch.float32, device='cuda')
    B = torch.randn(N, dtype=torch.float32, device='cuda')
    C = torch.empty_like(A)
    triton_kernel(A, B, C, N)
    torch.cuda.synchronize()
    expected = A + B
    if torch.allclose(C, expected):
        print("Triton kernel succeeded.")
    else:
        max_err = (C - expected).abs().max()
        print(f"Verification failed. error: {max_err}")