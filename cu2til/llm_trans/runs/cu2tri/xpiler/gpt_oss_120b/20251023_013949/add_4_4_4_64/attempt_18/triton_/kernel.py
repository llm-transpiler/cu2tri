# Triton implementation of the CUDA elementwise addition kernel.
# Requires: torch, triton (tested with Triton 3.4.0) on an NVIDIA H800 GPU.

import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: mirrors the original CUDA kernel logic.
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A, B, C, BLOCK_SIZE: tl.constexpr):
    """
    Compute C[i] = A[i] + B[i] for i < 4096.
    The kernel is launched with a block size of 1024 threads,
    matching the __launch_bounds__(1024) of the CUDA version.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices
    mask = offsets < 4096                       # replicate CUDA bound check

    # Load with mask (out‑of‑bounds loads return 0.0, matching CUDA undefined behavior)
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)

    # Store the result under the same mask
    tl.store(C + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function: entry point with the same signature as the original CUDA wrapper.
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Launches the Triton kernel that behaves identically to the original CUDA kernel.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32) on CUDA device.
    B : torch.Tensor
        Input tensor (float32) on CUDA device.
    C : torch.Tensor
        Output tensor (float32) on CUDA device.
    size : int
        Logical size of the vectors (used only for grid calculation, exactly as in the CUDA wrapper).
    """
    # ------------------------------------------------------------------
    # Argument validation (mirrors typical CUDA expectations)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"

    # ------------------------------------------------------------------
    # Grid configuration: identical to the CUDA launch configuration
    # ------------------------------------------------------------------
    BLOCK_SIZE = 1024
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )   # 1‑D grid

    # ------------------------------------------------------------------
    # Kernel launch.
    # num_warps=32 yields 1024 threads per block (32 warps * 32 threads/warp)
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE=BLOCK_SIZE, num_warps=32)


# ----------------------------------------------------------------------
# Example usage (can be removed when integrating into a larger code base)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a problem size larger than the kernel's hard‑coded bound to illustrate the behavior
    size = 5000
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    # Launch the Triton kernel
    triton_kernel(A, B, C, size)

    # Verify correctness for the first 4096 elements (the only region the kernel writes to)
    torch.testing.assert_allclose(C[:4096], A[:4096] + B[:4096])
    print("Triton kernel executed successfully and produced correct results for indices < 4096.")