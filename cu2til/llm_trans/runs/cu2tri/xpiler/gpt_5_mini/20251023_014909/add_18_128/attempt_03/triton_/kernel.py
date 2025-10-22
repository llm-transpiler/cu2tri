import torch
import triton
import triton.language as tl

# Triton kernel implementation — must be named exactly as requested.
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, BLOCK: tl.constexpr):
    """
    Each Triton program is responsible for BLOCK consecutive elements starting at
    pid * BLOCK. This mirrors the CUDA kernel having 1024 threads per block.
    The original CUDA kernel only writes when the global index < 2304; we
    reproduce that exact behavior here.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)            # global offsets handled by this program
    # Exact bound check from the original CUDA kernel:
    mask = offs < 2304

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(T_add_ptr + offs, c, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Entry-point wrapper with the same parameter signature as the original CUDA wrapper:
      triton_kernel(float *A, float *B, float *C, int size)

    A, B, C are expected to be contiguous torch.cuda.FloatTensor (float32).
    'size' is used only to compute the grid size (same as the CUDA wrapper).
    The kernel itself still only writes indices < 2304 to match the original behavior.
    """
    # Basic checks to keep behavior predictable and match the CUDA assumptions.
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor (CUDA tensors expected).")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors.")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must be float32 tensors.")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise ValueError("A, B, C must be contiguous (call .contiguous() before passing if needed).")
    if not isinstance(size, int):
        size = int(size)

    # Mirror the CUDA configuration: blockSize = 1024 threads
    BLOCK = 1024
    # numBlocks = (size + 1024 - 1) / 1024  (integer ceil division)
    num_blocks = (size + BLOCK - 1) // BLOCK
    if num_blocks <= 0:
        return  # Nothing to launch (matches CUDA behavior of computing 0 blocks)

    grid = (num_blocks,)
    # Launch Triton kernel. BLOCK is provided as a compile-time constant.
    _triton_kernel_impl[grid](A, B, C, BLOCK=BLOCK)


if __name__ == "__main__":
    # Quick self-test to validate behavior matches the CUDA kernel:
    device = torch.device("cuda")
    size = 4096
    A = torch.randn(size, device=device, dtype=torch.float32)
    B = torch.randn(size, device=device, dtype=torch.float32)
    C = torch.empty(size, device=device, dtype=torch.float32)

    # Run the Triton kernel (wrapper)
    triton_kernel(A, B, C, size)

    # The original CUDA kernel only writes indices < 2304
    expected = A[:2304] + B[:2304]
    if torch.allclose(C[:2304], expected):
        print("OK: First 2304 elements match A+B (as in original CUDA kernel).")
    else:
        print("ERROR: Mismatch in the first 2304 elements.")