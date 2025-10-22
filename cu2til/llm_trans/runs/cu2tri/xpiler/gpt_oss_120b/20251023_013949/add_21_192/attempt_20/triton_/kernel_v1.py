import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr,, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton implementation of elementwise addition: C[i] = A[i] + B[i] for i < size.
    Mirrors the behavior of the original CUDA kernel.
    """
    # Validate device and dtype
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be on CUDA device.")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise RuntimeError("All tensors must be of type torch.float32.")
    if size < 0:
        raise ValueError("size must be non‑negative.")
    # Early exit for empty work
    if size == 0:
        return
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    # Verify sufficient allocation
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError("Tensor sizes must be at least 'size'.")
    # Kernel launch configuration
    BLOCK_SIZE = 1024
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the Triton kernel
    _triton_kernel_impl[(num_blocks,)](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()

if __name__ == "__main__":
    # Simple correctness test
    torch.manual_seed(0)
    size = 4032  # matches the static bound in the original CUDA kernel
    A = torch.randn(size, dtype=torch.float32, device="cuda")
    B = torch.randn(size, dtype=torch.float32, device="cuda")
    C = torch.empty_like(A)
    triton_kernel(A, B, C, size)
    expected = A + B
    if not torch.allclose(C, expected, atol=1e-6):
        max_err = (C - expected).abs().max()
        raise AssertionError(f"Result mismatch! Max error: {max_err}")
    print("Triton kernel produced correct results.")