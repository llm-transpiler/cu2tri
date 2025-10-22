import torch
import triton
import triton.language as tl

# Triton kernel implementation (must be named exactly as requested)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK: tl.constexpr):
    """
    Each Triton program processes a contiguous window of length BLOCK.
    This mirrors the CUDA kernel which uses blocks of 1024 threads.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK, dtype=tl.int32)
    mask = offs < N
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offs, c, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Entry-point wrapper with the same signature as the original CUDA wrapper:
      triton_kernel(float *A, float *B, float *C, int size)

    Behavior:
      - Accepts torch CUDA tensors A, B, C (any shape); they will be flattened internally.
      - `size` may be a Python int or a 0-d torch tensor; it specifies how many elements to process.
      - Launches a Triton kernel with BLOCK=1024 to perform C[i] = A[i] + B[i] for i in [0, size).
    """
    # Normalize and validate size
    if isinstance(size, torch.Tensor):
        if size.numel() != 1:
            raise TypeError("`size` tensor must be a scalar (0-d) tensor")
        size = int(size.item())
    elif not isinstance(size, int):
        raise TypeError("`size` must be an int or a 0-d torch tensor containing an int")

    if size < 0:
        raise ValueError("`size` must be non-negative")

    # Validate tensors
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor instances")

    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, and C must be CUDA tensors")

    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise TypeError("A, B, and C must be float32 tensors")

    # Flatten tensors to 1-D (mirrors pointer semantics of the original CUDA kernel)
    A_flat = A.contiguous().view(-1)
    B_flat = B.contiguous().view(-1)
    C_flat = C.contiguous().view(-1)

    if A_flat.numel() < size or B_flat.numel() < size or C_flat.numel() < size:
        raise ValueError("A, B, and C must each have at least `size` elements")

    # Early exit if nothing to do (matches CUDA behavior of 0 blocks)
    if size == 0:
        return

    # Match the CUDA block size of 1024 threads
    BLOCK = 1024
    num_programs = (size + BLOCK - 1) // BLOCK
    grid = (num_programs,)

    # Launch Triton kernel on the current CUDA stream
    stream = torch.cuda.current_stream().cuda_stream
    _triton_kernel_impl[grid](A_flat, B_flat, C_flat, size, BLOCK=BLOCK, stream=stream)