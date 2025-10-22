import torch
import triton
import triton.language as tl

# Triton kernel implementation (must be named exactly as requested)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK: tl.constexpr):
    """
    A_ptr, B_ptr, C_ptr - device pointers (provided by passing torch tensors)
    size - int (present to match original signature; not used by internal bound check)
    BLOCK - number of elements handled per program instance (tl.constexpr)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)               # vector of indices handled by this program
    # Match the original CUDA kernel's hard-coded bound of 2304
    bound = 2304
    mask = offs < bound                                   # boolean mask
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(C_ptr + offs, a + b, mask=mask)


# Wrapper entry-point (must be named exactly as requested and keep same signature semantics)
def triton_kernel(A, B, C, size):
    """
    Entry point that mirrors the original CUDA kernel launcher:
      - A, B, C : torch.cuda.FloatTensor (1D)
      - size    : int (used for grid computation, as in the original CUDA launcher)
    Behavior is identical to the original CUDA code:
      - block size = 1024
      - numBlocks = (size + 1024 - 1) // 1024
      - the device kernel writes C[idx] = A[idx] + B[idx] only for idx < 2304 (constant),
        matching the original kernel's conditional.
    """
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor on CUDA device")

    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors")

    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must be float32 tensors")

    if not (A.ndim == 1 and B.ndim == 1 and C.ndim == 1):
        raise ValueError("A, B, C must be 1D tensors (contiguous linear buffers)")

    # Ensure size is an int for grid calculation
    size = int(size)

    BLOCK = 1024
    num_blocks = (size + BLOCK - 1) // BLOCK if size > 0 else 0
    if num_blocks == 0:
        return

    grid = (num_blocks,)

    # Launch Triton kernel. Pass `size` to mirror original signature (even though kernel uses hard-coded 2304).
    _triton_kernel_impl[grid](A, B, C, size, BLOCK=BLOCK)


# Optional simple smoke test when run as a script
if __name__ == "__main__":
    # Prepare test tensors
    N = 2500
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    C = torch.empty(N, device="cuda", dtype=torch.float32)

    # Call the Triton wrapper (signature matches original: A, B, C, size)
    triton_kernel(A, B, C, N)

    # Verify results for indices < 2304 (original kernel only writes those)
    bound = 2304
    expected = A + B
    max_diff = (C[:bound] - expected[:bound]).abs().max().item()
    print("max diff for indices [0..2303]:", max_diff)