import torch
import triton
import triton.language as tl

# Constant copied from the original CUDA kernel check
_LIMIT = 2304

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, BLOCK: tl.constexpr):
    """
    Triton kernel implementing:
      if (global_idx < 2304) T_add[global_idx] = A[global_idx] + B[global_idx];
    Each program handles BLOCK elements.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < _LIMIT
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Entry-point wrapper that mimics the original CUDA host function signature:
      cuda_kernel(float *A, float *B, float *C, int size)

    Parameters:
      A, B, C : torch.cuda.FloatTensor (1D)
      size     : int (used to compute grid dim exactly like the original)
    Behavior:
      - computes numBlocks = (size + 1024 - 1) // 1024
      - launches _triton_kernel_impl with BLOCK=1024
      - the device kernel itself uses the identical check `idx < 2304`
        as in the original CUDA kernel.
    """
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor instances on CUDA")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must be float32 tensors")
    if A.dim() != 1 or B.dim() != 1 or C.dim() != 1:
        raise ValueError("A, B, C must be 1-D tensors (flat arrays)")

    BLOCK = 1024
    # Compute numBlocks exactly as in the original CUDA host code
    num_blocks = (int(size) + BLOCK - 1) // BLOCK
    if num_blocks <= 0:
        return

    grid = (num_blocks,)

    # Launch Triton kernel. Pass tensors directly; BLOCK is a compile-time constant.
    _triton_kernel_impl[grid](A, B, C, BLOCK=BLOCK)


# Optional test when run as script
if __name__ == "__main__":
    # Simple self-check to ensure behavior matches A + B for indices < 2304
    device = "cuda"
    size = 2304  # same as the constant used in the kernel for full coverage
    A = torch.arange(size, dtype=torch.float32, device=device)
    B = torch.arange(size, dtype=torch.float32, device=device) * 2.0
    C = torch.empty(size, dtype=torch.float32, device=device)

    triton_kernel(A, B, C, size)

    # Validate
    expected = A + B
    if not torch.allclose(C, expected):
        # If mismatch, raise for visibility
        diff = (C - expected).abs().max().item()
        raise RuntimeError(f"Mismatch after Triton kernel launch. max diff = {diff}")
    else:
        print("Triton kernel produced correct results for the test.")