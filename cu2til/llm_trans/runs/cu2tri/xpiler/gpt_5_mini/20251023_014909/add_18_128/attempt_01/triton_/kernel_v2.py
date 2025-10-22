import torch
import triton
import triton.language as tl

# Triton kernel implementation: element-wise add A + B -> T_add
# Must be named exactly as requested.
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, check_n, BLOCK: tl.constexpr):
    """
    A_ptr, B_ptr, T_add_ptr : pointers to 1D float32 tensors
    check_n : runtime integer used for the bounds check (the original CUDA kernel used the hard-coded 2304)
    BLOCK : compile-time block size (should be 1024 to match the CUDA kernel)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)            # removed dtype kw (not supported in this Triton version)
    mask = offs < check_n
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Wrapper entry point that matches the CUDA kernel signature:
      triton_kernel(A, B, C, size)
    where A, B, C are torch.cuda.FloatTensor and size is an int.

    This mirrors the original CUDA behavior:
      - Uses block size 1024
      - Computes numBlocks = (size + 1024 - 1) // 1024
      - The kernel performs a bounds check against the constant 2304 (replicating the original kernel)
    """
    # Basic validations to keep types consistent with original CUDA floats
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must be torch.float32 tensors")
    if not isinstance(size, int):
        raise TypeError("size must be an int")
    # Ensure 1D contiguous tensors (matches CUDA pointer semantics)
    if A.dim() != 1 or B.dim() != 1 or C.dim() != 1:
        raise ValueError("A, B, C must be 1-D tensors")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()

    # Match CUDA launch configuration
    BLOCK = 1024
    num_blocks = (size + BLOCK - 1) // BLOCK
    grid = (num_blocks,)

    # Launch Triton kernel.
    # To replicate original CUDA kernel's behavior exactly, pass 2304 as the check value.
    _triton_kernel_impl[grid](A, B, C, 2304, BLOCK=BLOCK)


# Optional self-test when run as a script
if __name__ == "__main__":
    # Example usage consistent with the original CUDA where size == 2304
    size = 2304
    A = torch.randn(size, device='cuda', dtype=torch.float32)
    B = torch.randn(size, device='cuda', dtype=torch.float32)
    C = torch.empty(size, device='cuda', dtype=torch.float32)

    triton_kernel(A, B, C, size)
    torch.cuda.synchronize()

    expected = A + B
    if not torch.allclose(C, expected):
        diff_max = (C - expected).abs().max().item()
        raise RuntimeError(f"Result mismatch (max abs diff = {diff_max})")
    else:
        print("Triton kernel succeeded: C == A + B")