import torch
import triton
import triton.language as tl

# Triton kernel implementation (must be named exactly as requested)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK: tl.constexpr):
    """
    Each Triton program handles a contiguous window of length BLOCK.
    This mirrors the CUDA kernel which uses blocks of 1024 threads.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)              # indices handled by this program
    mask = offsets < size                                    # bounds check to avoid OOB accesses

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)       # load A[offsets] with masking
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)       # load B[offsets] with masking
    c = a + b                                                # compute
    tl.store(C_ptr + offsets, c, mask=mask)                  # store result to C[offsets]


def triton_kernel(A, B, C, size):
    """
    Wrapper entry point named `triton_kernel` with the same parameter signature
    semantics as the original CUDA kernel: (float *A, float *B, float *C, int size).

    Requirements:
      - A, B, C must be torch.cuda.FloatTensor
      - A, B, C must be 1-D and contiguous
      - number of elements in each must be >= size
    """
    # Validate size argument
    if not isinstance(size, int):
        # Allow torch scalar input as well (e.g., torch.tensor(4096, device='cpu'))
        if isinstance(size, torch.Tensor) and size.numel() == 1:
            size = int(size.item())
        else:
            raise TypeError("`size` must be an int or a 0-d torch tensor containing an int")

    # Basic validations to ensure consistent behaviour with the CUDA kernel
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor instances on CUDA")

    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, and C must be CUDA tensors")

    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise TypeError("A, B, and C must be float32 tensors")

    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise ValueError("A, B, and C must be contiguous (call .contiguous() before passing)")

    if not (A.ndim == B.ndim == C.ndim == 1):
        raise ValueError("A, B, and C must be 1-D tensors (flatten them if needed)")

    if not (A.numel() >= size and B.numel() >= size and C.numel() >= size):
        raise ValueError("A, B, and C must each have at least `size` elements")

    # Match the CUDA block size of 1024 threads
    BLOCK = 1024
    grid = ( (size + BLOCK - 1) // BLOCK, )

    # Launch Triton kernel on the current CUDA stream
    _triton_kernel_impl[grid](A, B, C, size, BLOCK=BLOCK)