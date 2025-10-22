import math
import torch
import triton
import triton.language as tl

# Triton kernel implementation (must be named exactly as requested)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK: tl.constexpr):
    """
    Triton kernel that computes C[i] = A[i] + B[i] for a block of elements.
    BLOCK is a compile-time constant that represents the logical block size
    (matches the CUDA blockDim.x used in the original kernel).
    """
    pid = tl.program_id(0)
    # Compute the global offsets handled by this program (block)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(T_add_ptr + offs, c, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Wrapper that launches the Triton kernel. Signature matches the CUDA wrapper:
      triton_kernel(A, B, C, size)

    Parameters:
      A (torch.Tensor): input tensor (device='cuda', dtype=torch.float32)
      B (torch.Tensor): input tensor (device='cuda', dtype=torch.float32)
      C (torch.Tensor): output tensor (device='cuda', dtype=torch.float32)
      size (int): number of elements to process
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available to run the Triton kernel.")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors.")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise ValueError("A, B, C must be torch.float32 tensors.")
    size = int(size)
    if size <= 0:
        return

    # Match the CUDA block size used in the original kernel
    BLOCK = 960
    # Compute number of blocks (grid size) the same way as the CUDA wrapper
    num_blocks = (size + BLOCK - 1) // BLOCK
    grid = (num_blocks,)

    # Launch Triton kernel. BLOCK is passed as a compile-time constant.
    # We do not force a specific num_warps here; Triton will choose defaults,
    # but BLOCK matches the logical CUDA blockDim.x.
    _triton_kernel_impl[grid](A, B, C, size, BLOCK=BLOCK)


# Optional quick test when running this file directly.
if __name__ == "__main__":
    # Example usage: vector add for N elements
    N = 10_000
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    # Launch Triton wrapper (same signature as the CUDA wrapper)
    triton_kernel(A, B, C, N)

    # Verify correctness
    if not torch.allclose(C[:N], A + B):
        max_err = (C[:N] - (A + B)).abs().max().item()
        raise RuntimeError(f"Result mismatch (max abs error = {max_err})")
    else:
        print("triton_kernel result is correct for N =", N)