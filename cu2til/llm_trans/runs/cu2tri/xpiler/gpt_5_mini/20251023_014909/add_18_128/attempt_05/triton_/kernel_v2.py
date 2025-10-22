import torch
import triton
import triton.language as tl

# Triton kernel must be named exactly as requested
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, SIZE, BLOCK: tl.constexpr):
    """
    Each Triton program handles BLOCK consecutive elements:
      offs = program_id(0) * BLOCK + arange(0, BLOCK)
    and performs C[offs] = A[offs] + B[offs] with masking for out-of-bounds.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK, dtype=tl.int32)
    mask = offs < SIZE
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offs, c, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Wrapper entry point with the same signature as the original CUDA kernel:
      triton_kernel(A, B, C, size)

    A, B, C: torch.Tensor on CUDA with dtype=torch.float32 (will be converted if needed)
    size: int
    """
    # Basic validation
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor instances on a CUDA device")

    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors")

    # Enforce float32 dtype (matches original float* kernel)
    if A.dtype != torch.float32:
        A = A.to(dtype=torch.float32)
    if B.dtype != torch.float32:
        B = B.to(dtype=torch.float32)
    if C.dtype != torch.float32:
        C = C.to(dtype=torch.float32)

    # Ensure size is a Python int
    size = int(size)
    if size <= 0:
        return  # nothing to do

    # Check buffers are large enough
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError(f"Buffers are too small for requested size={size}. "
                         f"Got sizes A={A.numel()}, B={B.numel()}, C={C.numel()}")

    # Prepare contiguous 1-D views. If C is non-contiguous we will copy the result back.
    A_ = A if A.is_contiguous() else A.contiguous()
    B_ = B if B.is_contiguous() else B.contiguous()
    if C.is_contiguous():
        C_ = C
        copy_back = False
    else:
        C_ = C.contiguous()
        copy_back = True

    # Flatten to 1D (kernel treats memory as flat float* as in CUDA)
    A_ = A_.view(-1)
    B_ = B_.view(-1)
    C_ = C_.view(-1)

    BLOCK = 1024  # match CUDA block size in the original code
    num_programs = (size + BLOCK - 1) // BLOCK
    # Triton expects a tuple for grid size
    grid = (num_programs,)

    # Launch Triton kernel
    _triton_kernel_impl[grid](A_, B_, C_, size, BLOCK=BLOCK)

    # If we used a temporary contiguous output buffer, copy results back into original C
    if copy_back:
        C.copy_(C_)


if __name__ == "__main__":
    # Small self-test to verify correctness (runs only when executed as a script)
    if not torch.cuda.is_available():
        raise SystemError("CUDA is required to run this test.")

    # Example size (original CUDA example used size=2304)
    size = 2304
    a = torch.randn(size, dtype=torch.float32, device='cuda')
    b = torch.randn(size, dtype=torch.float32, device='cuda')
    c = torch.empty(size, dtype=torch.float32, device='cuda')

    # Run Triton kernel
    triton_kernel(a, b, c, size)

    # Validate
    expected = a + b
    if not torch.allclose(c, expected):
        raise AssertionError("Results do not match expected (a + b).")
    else:
        print("Triton kernel test passed.")