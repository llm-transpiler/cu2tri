import torch
import triton
import triton.language as tl

# Triton kernel implementation (must be named exactly as requested)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK: tl.constexpr):
    """
    Triton kernel that mirrors the CUDA kernel's behavior.
    Note: It intentionally uses the same fixed bound check (< 4032) as in the CUDA kernel.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # Replicate the CUDA kernel's bound check (literal 4032)
    mask = offs < 4032
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


# Wrapper entry point (must be named exactly as requested and have same parameter signature)
def triton_kernel(A, B, C, size):
    """
    Entry point that mirrors the original CUDA kernel signature:
      triton_kernel(A, B, C, size)

    A, B, C: torch.cuda.FloatTensor (1-D or broadcastable to 1-D), dtype=torch.float32
    size: int (used to compute grid size, just like the original CUDA wrapper)
    """
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor) or not isinstance(C, torch.Tensor):
        raise TypeError("A, B, C must be torch.Tensor")
    if not A.is_cuda or not B.is_cuda or not C.is_cuda:
        raise ValueError("A, B, C must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("A, B, C must be torch.float32")

    # Flatten to 1-D if needed (CUDA kernel treats inputs as linear buffers)
    if A.dim() != 1:
        A = A.reshape(-1)
    if B.dim() != 1:
        B = B.reshape(-1)
    if C.dim() != 1:
        C = C.reshape(-1)

    # Ensure contiguous layout for Triton pointer passing
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # Mirror CUDA launch configuration
    blockSize = 1024
    numBlocks = (int(size) + blockSize - 1) // blockSize

    # Launch Triton kernel
    _triton_kernel_impl[numBlocks](A, B, C, size, BLOCK=blockSize)


# Example usage / simple test
if __name__ == "__main__":
    # Use size consistent with the original kernel bound for a straightforward test
    size = 4032
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty(size, device="cuda", dtype=torch.float32)

    triton_kernel(A, B, C, size)
    torch.cuda.synchronize()

    # Validate correctness
    max_err = (C - (A + B)).abs().max().item()
    print("Max absolute error:", max_err)