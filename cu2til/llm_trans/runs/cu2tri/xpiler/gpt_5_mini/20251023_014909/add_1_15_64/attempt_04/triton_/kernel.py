import torch
import triton
import triton.language as tl

# Triton kernel implementation (must be named exactly as requested)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLK: tl.constexpr, VEC: tl.constexpr):
    """
    Triton kernel that computes T_add[i] = A[i] + B[i] for a logical block of size BLK.
    VEC must be a power-of-two and >= BLK; tl.arange is called with VEC to satisfy Triton's requirement.
    Each program_id(0) handles BLK elements starting at pid * BLK.
    """
    pid = tl.program_id(0)
    # vector indices 0..VEC-1 (VEC must be power-of-two)
    idx = tl.arange(0, VEC)
    start = pid * BLK
    offs = start + idx
    # valid if idx is within the logical block (idx < BLK) AND global offset < size
    mask = (idx < BLK) & (offs < size)
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
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise ValueError("A, B, C must be torch.Tensor objects.")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors.")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise ValueError("A, B, C must be torch.float32 tensors.")

    size = int(size)
    if size <= 0:
        return

    # Match the CUDA block size used in the original kernel (logical block size)
    BLOCK = 960

    # Triton requires arange's range to be a power-of-two. Choose VEC = next power of two >= BLOCK.
    VEC = 1 << ((BLOCK - 1).bit_length())

    # Compute number of blocks (grid size) same as CUDA wrapper
    num_blocks = (size + BLOCK - 1) // BLOCK
    grid = (num_blocks,)

    # Launch Triton kernel. BLK and VEC are passed as compile-time constants.
    _triton_kernel_impl[grid](A, B, C, size, BLK=BLOCK, VEC=VEC)


# Optional quick test when running this file directly.
if __name__ == "__main__":
    N = 10_000
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    triton_kernel(A, B, C, N)

    if not torch.allclose(C[:N], A + B):
        max_err = (C[:N] - (A + B)).abs().max().item()
        raise RuntimeError(f"Result mismatch (max abs error = {max_err})")
    else:
        print("triton_kernel result is correct for N =", N)