import torch
import triton
import triton.language as tl

# Triton kernel implementation: element-wise add A + B -> T_add
# Named exactly as requested.
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, check_n, BLOCK: tl.constexpr):
    """
    A_ptr, B_ptr, T_add_ptr : pointers to 1D float32 tensors
    check_n : integer used for the bounds check in the kernel (matches original CUDA's hard-coded 2304)
    BLOCK : compile-time block size (should be 1024 to match the CUDA kernel)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK, dtype=tl.int32)
    mask = offs < check_n  # replicate original CUDA kernel's bounds check (2304)
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Wrapper entry point that matches the CUDA kernel signature:
      triton_kernel(A, B, C, size)
    where A, B, C are torch.cuda.FloatTensor and size is an int.
    This configures the grid/block and launches the Triton kernel.

    It intentionally mirrors the original CUDA behavior:
      - Uses block size 1024
      - Computes numBlocks = (size + 1024 - 1) // 1024
      - The kernel itself uses a hard-coded bound check of 2304 (replicating the original kernel)
    """
    # Basic checks to keep types consistent with the original CUDA floats
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must be torch.float32 tensors")
    if not isinstance(size, int):
        raise TypeError("size must be an int")

    # Match CUDA launch configuration
    BLOCK = 1024
    num_blocks = (size + BLOCK - 1) // BLOCK
    grid = (num_blocks,)

    # Launch Triton kernel.
    # Note: to replicate the original CUDA kernel's behavior exactly, we pass 2304 as the check value.
    _triton_kernel_impl[grid](A, B, C, 2304, BLOCK=BLOCK)


# Optional self-test when run as a script
if __name__ == "__main__":
    # Example usage; consistent with original CUDA example where size == 2304
    size = 2304
    A = torch.randn(size, device='cuda', dtype=torch.float32)
    B = torch.randn(size, device='cuda', dtype=torch.float32)
    C = torch.empty(size, device='cuda', dtype=torch.float32)

    triton_kernel(A, B, C, size)
    torch.cuda.synchronize()

    # Validate results
    expected = A + B
    if not torch.allclose(C, expected):
        # If mismatch, raise an error for visibility
        diff_max = (C - expected).abs().max().item()
        raise RuntimeError(f"Result mismatch (max abs diff = {diff_max})")
    else:
        print("Triton kernel succeeded: C == A + B")