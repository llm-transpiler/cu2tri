import torch
import triton
import triton.language as tl

# Triton kernel implementation (must be named exactly as requested)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK: tl.constexpr):
    """
    Triton kernel that performs elementwise A + B -> T_add.
    BLOCK is a compile-time constant (threads per block). We follow the
    canonical, correct mapping that matches the intended CUDA behavior:
      global_idx = program_id(0) * BLOCK + range(0, BLOCK)
    and we guard loads/stores with a mask (offs < size).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(T_add_ptr + offs, c, mask=mask)


# Wrapper entry point (must be named exactly as requested and keep same signature)
def triton_kernel(A, B, C, size):
    """
    Entry point mirroring the original CUDA wrapper signature:
      triton_kernel(float *A, float *B, float *C, int size)

    Parameters:
      A, B, C : torch.cuda.FloatTensor (1-D) - device tensors
      size    : int - number of elements to process

    Notes:
      - This wrapper launches a Triton kernel with BLOCK = 320 threads per block,
        and number of blocks = ceil(size / BLOCK), matching the CUDA wrapper's grid config.
      - The kernel writes C[i] = A[i] + B[i] for i in [0, size).
    """
    # Parameters / configuration
    BLOCK = 320  # must match the original CUDA block size

    # Basic type and device checks to ensure correct usage
    if not (torch.is_tensor(A) and torch.is_tensor(B) and torch.is_tensor(C)):
        raise TypeError("A, B, C must be torch tensors on CUDA with dtype=torch.float32")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("A, B, C must be float32 tensors")
    if not A.is_cuda or not B.is_cuda or not C.is_cuda:
        raise ValueError("A, B, C must be CUDA tensors")
    if size < 0:
        raise ValueError("size must be non-negative")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError("A, B, and C must have at least 'size' elements")

    # Ensure tensors are contiguous for best performance with Triton
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    # Grid configuration: same as CUDA wrapper
    num_blocks = (size + BLOCK - 1) // BLOCK
    # Launch Triton kernel. BLOCK is passed as a compile-time constant.
    _triton_kernel_impl[(num_blocks,)](A, B, C, size, BLOCK=BLOCK)

    # Synchronize to match CUDA's default asynchronous behavior if caller expects results immediately.
    # (Optional: remove to allow asynchronous operation.)
    torch.cuda.synchronize()


# Optional simple test when run as a script
if __name__ == "__main__":
    # Small self-test to validate correctness
    size = 1024
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    triton_kernel(a, b, c, size)

    # Validate result
    expected = a + b
    torch.cuda.synchronize()
    if torch.allclose(c, expected):
        print("Test passed: Triton kernel produced correct results.")
    else:
        # Print max difference for debugging
        diff = (c - expected).abs().max().item()
        print(f"Test failed: max abs difference = {diff}")