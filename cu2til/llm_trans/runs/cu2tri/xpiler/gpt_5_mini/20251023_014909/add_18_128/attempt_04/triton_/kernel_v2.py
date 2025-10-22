import torch
import triton
import triton.language as tl

# Keep the same hard-coded limit present in the original CUDA kernel
_LIMIT = 2304

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, BLOCK: tl.constexpr):
    """
    Triton implementation of the original CUDA kernel:
      if (global_idx < 2304) T_add[global_idx] = A[global_idx] + B[global_idx];
    Each program instance handles BLOCK consecutive elements.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < _LIMIT
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Host wrapper matching the original CUDA host signature:
      cuda_kernel(float *A, float *B, float *C, int size)

    Parameters:
      A, B, C : torch.Tensor (on CUDA). Can be of any shape; will be flattened.
      size     : int, used to compute grid dims exactly as the original.
    Behavior:
      - numBlocks = (size + 1024 - 1) // 1024
      - launches _triton_kernel_impl with BLOCK=1024
      - the device kernel uses the identical check `idx < 2304`
        as in the original CUDA kernel.
    """
    # Basic type/device checks
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor instances")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must be float32 tensors")

    # Flatten inputs to 1-D contiguous tensors (mimics pointer arithmetic on flat arrays)
    A_flat = A.contiguous().view(-1)
    B_flat = B.contiguous().view(-1)
    C_flat = C.contiguous().view(-1)

    # Compute grid exactly like original CUDA host code
    BLOCK = 1024
    num_blocks = (int(size) + BLOCK - 1) // BLOCK
    if num_blocks <= 0:
        return

    grid = (num_blocks,)

    # Launch Triton kernel. BLOCK is passed as a compile-time constant.
    _triton_kernel_impl[grid](A_flat, B_flat, C_flat, BLOCK=BLOCK)


# Optional self-test when run as a script
if __name__ == "__main__":
    # Create test data on CUDA
    device = "cuda"
    # Use a shape that demonstrates non-1D inputs (e.g., 18 x 128 = 2304)
    H, W = 18, 128
    size = H * W
    A = torch.arange(size, dtype=torch.float32, device=device).reshape(H, W)
    B = (torch.arange(size, dtype=torch.float32, device=device) * 2.0).reshape(H, W)
    C = torch.empty((H, W), dtype=torch.float32, device=device)

    # Launch Triton wrapper
    triton_kernel(A, B, C, size)

    # Validate results for indices < 2304 (which covers all elements here)
    expected = (A + B).reshape(-1)
    got = C.reshape(-1)
    if not torch.allclose(got, expected):
        diff = (got - expected).abs().max().item()
        raise RuntimeError(f"Mismatch after Triton kernel launch. max diff = {diff}")
    else:
        print("Triton kernel produced correct results for the test.")