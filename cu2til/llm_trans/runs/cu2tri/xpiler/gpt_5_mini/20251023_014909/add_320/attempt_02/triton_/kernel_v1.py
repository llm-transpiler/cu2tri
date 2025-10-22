import torch
import triton
import triton.language as tl

# Triton kernel: named exactly as required
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, n_elements, BLOCK: tl.constexpr):
    """
    Triton implementation of elementwise addition.
    A_ptr, B_ptr, C_ptr : pointers to float32 arrays (passed as torch tensors)
    n_elements          : number of valid elements to process (int)
    BLOCK               : block size (constexpr)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Entry-point wrapper that mirrors the CUDA host wrapper:
      cuda_kernel(float *A, float *B, float *C, int size)

    Parameters:
      A, B, C : torch.cuda.FloatTensor (1D) - inputs and output
      size    : int - number of elements to process

    This function configures the grid and launches the Triton kernel.
    """
    # Basic checks to keep behavior predictable / consistent with CUDA expectations
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must be float32 tensors")
    if size < 0:
        raise ValueError("size must be non-negative")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError("A, B, C must have at least 'size' elements")

    # Match the CUDA block size from the original code
    BLOCK = 320
    num_blocks = (size + BLOCK - 1) // BLOCK
    grid = (num_blocks,)

    # Launch Triton kernel. BLOCK is passed as a constexpr.
    _triton_kernel_impl[grid](A, B, C, size, BLOCK=BLOCK)
    # Ensure the kernel has finished before returning (mirrors CUDA semantics)
    torch.cuda.synchronize()


if __name__ == "__main__":
    # Simple test to verify correctness (directly executable)
    size = 1000
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty(size, device="cuda", dtype=torch.float32)

    triton_kernel(A, B, C, size)

    # Validate result
    expected = A + B
    max_err = (C - expected).abs().max().item()
    print(f"max error: {max_err:.6e}")