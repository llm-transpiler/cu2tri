import torch
import triton
import triton.language as tl

# Triton kernel implementation (must be named _triton_kernel_impl)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, BLOCK: tl.constexpr):
    """
    Triton kernel that mirrors the original CUDA kernel behavior:
    Each program (block) handles BLOCK contiguous elements.
    The original CUDA kernel used a hard-coded bound of 4032 inside the kernel;
    we preserve that exact bound here to match behavior.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # preserve the exact bound check from the original CUDA kernel
    mask = offs < 4032
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(T_add_ptr + offs, c, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Entry-point wrapper mirroring the CUDA wrapper:
      void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters:
    - A, B, C: torch.cuda.FloatTensor (1D or larger). They must be on CUDA and dtype=float32.
               The kernel will write into C (same role as T_add in the CUDA kernel).
    - size: integer used to compute the number of blocks (kept identical to original wrapper logic).

    Notes:
    - The kernel uses a hard-coded in-kernel bound of 4032 (as in the original CUDA kernel).
    - Grid sizing is computed exactly as in the original CUDA wrapper:
        blockSize = 1024
        numBlocks = (size + 1024 - 1) // 1024
    - This function does not synchronize the device (same as original CUDA wrapper).
    """
    # Basic input validation to ensure correct types (keeps behavior safe for direct execution)
    if not (torch.is_tensor(A) and torch.is_tensor(B) and torch.is_tensor(C)):
        raise TypeError("A, B, C must be torch.Tensor on CUDA device")
    if not (A.device.type == "cuda" and B.device.type == "cuda" and C.device.type == "cuda"):
        raise ValueError("A, B, C must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must be torch.float32 dtype")

    # Ensure contiguous tensors for Triton kernels
    A_ = A.contiguous()
    B_ = B.contiguous()
    C_ = C.contiguous()

    BLOCK = 1024
    num_blocks = (int(size) + BLOCK - 1) // BLOCK
    if num_blocks <= 0:
        return

    # Launch Triton kernel with the same grid/block configuration logic
    _triton_kernel_impl[(num_blocks,)](A_, B_, C_, BLOCK=BLOCK)


if __name__ == "__main__":
    # Quick correctness smoke-test (only runs when executed as a script).
    # Requires CUDA + Triton + PyTorch to be available.
    if not torch.cuda.is_available():
        print("CUDA not available; aborting test.")
    else:
        # Allocate tensors. To fully exercise the kernel bound behavior, use size = 4032.
        size = 4032
        A = torch.randn(size, device="cuda", dtype=torch.float32)
        B = torch.randn(size, device="cuda", dtype=torch.float32)
        C = torch.empty(size, device="cuda", dtype=torch.float32)

        # Launch triton kernel
        triton_kernel(A, B, C, size)

        # Compare with expected result
        expected = A + B
        # synchronize to ensure kernel has finished for the check
        torch.cuda.synchronize()
        max_diff = (C - expected).abs().max().item()
        print(f"max absolute difference: {max_diff:.6e}")