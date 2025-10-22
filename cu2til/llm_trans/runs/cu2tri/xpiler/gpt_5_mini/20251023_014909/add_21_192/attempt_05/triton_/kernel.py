import math
import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise A + B -> T_add
# Named exactly as required by the task.
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(T_add_ptr + offsets, c, mask=mask)


# Wrapper with the same signature as the original CUDA entry point:
#   extern "C" void cuda_kernel(float *A, float *B, float *C, int size)
# Here: triton_kernel(A, B, C, size)
def triton_kernel(A, B, C, size):
    """
    Launch the Triton kernel that performs:
      for i in range(size):
        C[i] = A[i] + B[i]

    Parameters:
      A, B, C : torch.cuda.FloatTensor (1D or convertible to 1D contiguous)
      size    : int (number of elements to process)
    """
    # Cast/validate size
    size = int(size)

    # Basic validation to match the CUDA expectations (float* on device)
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor objects")
    if not (A.device.type == "cuda" and B.device.type == "cuda" and C.device.type == "cuda"):
        raise ValueError("A, B, C must all be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must have dtype torch.float32")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError("Tensors do not contain enough elements for the provided size")

    # Ensure contiguous memory layout (Triton expects contiguous buffers for pointer-style access)
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    # Match the CUDA kernel configuration: block size 1024 threads, grid = ceil(size / 1024)
    BLOCK = 1024
    num_blocks = (size + BLOCK - 1) // BLOCK
    if num_blocks == 0:
        return

    # Launch: one program instance per block (1D grid)
    # Pass BLOCK as a compile-time constant
    _triton_kernel_impl[(num_blocks,)](A, B, C, size, BLOCK=BLOCK)


# Example self-test when run as a script
if __name__ == "__main__":
    # Small test that reproduces the original kernel behavior (size 4032 used in original CUDA code)
    n = 4032
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    c = torch.empty(n, device="cuda", dtype=torch.float32)

    triton_kernel(a, b, c, n)

    # Validate
    expected = a + b
    if not torch.allclose(c, expected):
        print("ERROR: results do not match")
        # For debugging, you could print differences; kept minimal here.
        raise SystemExit(1)
    else:
        print("OK: Triton kernel produced correct results")