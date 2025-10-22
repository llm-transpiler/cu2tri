import torch
import triton
import triton.language as tl

# Triton kernel implementation: must be named exactly `_triton_kernel_impl`.
# It mirrors the original CUDA kernel logic: each "block" processes 1024 elements,
# and the kernel only writes elements with global index < 4032 (constant).
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK, dtype=tl.int32)
    # Replicate CUDA kernel's hard-coded bound check: index < 4032
    mask = offs < 4032
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(C_ptr + offs, a + b, mask=mask)


# Entry-point wrapper: must be named `triton_kernel` and have the same parameter
# signature as the original CUDA wrapper: (A, B, C, size).
def triton_kernel(A, B, C, size):
    """
    A, B, C : torch.cuda.FloatTensor (or convertible) 1D tensors.
    size    : int (used to configure the grid exactly as in the original CUDA wrapper)
    Note: The kernel itself performs a bound check against the constant 4032,
    matching the original CUDA kernel logic.
    """
    # Basic type/device checks and conversions to match expected pointer types.
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor) or not isinstance(C, torch.Tensor):
        raise TypeError("A, B, C must be torch.Tensor")
    if not A.is_cuda or not B.is_cuda or not C.is_cuda:
        raise ValueError("A, B, C must be CUDA tensors")
    # Ensure float32 dtype (original CUDA used float)
    if A.dtype != torch.float32:
        A = A.to(torch.float32)
    if B.dtype != torch.float32:
        B = B.to(torch.float32)
    if C.dtype != torch.float32:
        C = C.to(torch.float32)
    # Ensure contiguous memory for Triton
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    size = int(size)
    BLOCK = 1024  # exactly as in the CUDA block size
    # Compute number of blocks exactly like the original wrapper:
    num_blocks = (size + BLOCK - 1) // BLOCK
    if num_blocks <= 0:
        return

    # Launch Triton kernel. We pass BLOCK as a compile-time constant.
    # For performance on modern hardware, specify num_warps (tuneable). Using 8 as a sensible default.
    _triton_kernel_impl[(num_blocks,)](A, B, C, size, BLOCK=BLOCK, num_warps=8)


# Example / quick test to verify correctness when executing this file directly.
if __name__ == "__main__":
    # Prepare sample data
    N = 4032  # the CUDA kernel used a 4032 bound; using same for a straightforward test
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    c = torch.empty_like(a)

    # Call the Triton wrapper (same signature as CUDA wrapper)
    triton_kernel(a, b, c, N)

    # Validate results
    expected = a + b
    if not torch.allclose(c, expected):
        # If mismatch, raise an informative error
        diff = (c - expected).abs().max().item()
        raise RuntimeError(f"Mismatch between Triton kernel output and expected result. max diff = {diff}")
    else:
        print("triton_kernel produced correct results for N =", N)