import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: elementwise addition with a hardcoded bound of 4032.
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024  # Must match the CUDA __launch_bounds__(1024)

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton implementation of the original CUDA kernel.
    Parameters
    ----------
    A_ptr, B_ptr, C_ptr : pointers to the input and output float32 arrays.
    N : int
        Unused parameter kept for signature compatibility.
    BLOCK_SIZE : int (constexpr)
        Number of threads per block (must be 1024).
    """
    pid = tl.program_id(0)  # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices for this block
    mask = offsets < 4032  # hard‑coded bound from the CUDA kernel

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that mimics the original CUDA kernel launch.
    Arguments
    ---------
    A, B, C : torch.Tensor
        Float32 CUDA tensors with identical shapes.
    size : int
        Logical size used only for grid calculation (mirrors the CUDA API).
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise RuntimeError("All tensors must be of type torch.float32")
    if A.shape != B.shape or A.shape != C.shape:
        raise RuntimeError("Input tensors must have the same shape")

    # Ensure contiguous memory layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # ------------------------------------------------------------------
    # Grid configuration (matches the CUDA launch configuration)
    # ------------------------------------------------------------------
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,               # N (unused inside the kernel)
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32        # 1024 threads = 32 warps
    )
    torch.cuda.synchronize()


# ----------------------------------------------------------------------
# Simple sanity check (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example size (>= 4032 to fully exercise the kernel)
    size = 5000
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    triton_kernel(A, B, C, size)

    # Verify the first 4032 elements (the only ones the kernel writes)
    expected = A[:4032] + B[:4032]
    if torch.allclose(C[:4032], expected):
        print("Success: Triton kernel matches expected results for the first 4032 elements.")
    else:
        diff = (C[:4032] - expected).abs().max()
        print(f"Mismatch detected. Max difference: {diff.item()}")