import triton
import triton.language as tl
import torch


# Triton kernel implementation. Must be named exactly `_triton_kernel_impl`.
# We use a blocked/chunked loop so we can support BLOCK=960 (as in the original CUDA)
# while mapping to a reasonable vector width (CHUNK = num_warps * 32).
@triton.jit(num_warps=8)
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    rng = tl.arange(0, CHUNK)

    # iterate over the BLOCK in chunks of CHUNK lanes
    # ensure we don't touch elements outside the block (i + rng < BLOCK)
    # and we don't touch elements >= size
    for i in range(0, BLOCK, CHUNK):
        offs = block_start + i + rng
        in_block = (i + rng) < BLOCK
        in_range = offs < size
        mask = in_block & in_range
        a = tl.load(A_ptr + offs, mask=mask, other=0.0)
        b = tl.load(B_ptr + offs, mask=mask, other=0.0)
        c = a + b
        tl.store(C_ptr + offs, c, mask=mask)


# Wrapper matching the CUDA kernel signature exactly:
# triton_kernel(float *A, float *B, float *C, int size)
def triton_kernel(A, B, C, size):
    """
    A, B, C: torch.Tensor on CUDA, dtype=torch.float32, 1-D (or will be flattened)
    size: number of elements to process (int)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Triton kernels require a CUDA device.")

    # Basic type/device/shape checks and preparations
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor) or not isinstance(C, torch.Tensor):
        raise TypeError("A, B, and C must be torch.Tensor")

    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("A, B, and C must be torch.float32")

    if not A.is_cuda or not B.is_cuda or not C.is_cuda:
        raise ValueError("A, B, and C must be on CUDA device")

    # Flatten to 1-D contiguous tensors (original CUDA kernel treats inputs as flat pointers)
    if A.dim() != 1 or not A.is_contiguous():
        A = A.contiguous().view(-1)
    else:
        A = A.view(-1)
    if B.dim() != 1 or not B.is_contiguous():
        B = B.contiguous().view(-1)
    else:
        B = B.view(-1)
    if C.dim() != 1 or not C.is_contiguous():
        C = C.contiguous().view(-1)
    else:
        C = C.view(-1)

    n = int(size)

    if A.numel() < n or B.numel() < n or C.numel() < n:
        raise ValueError("Buffers A, B, and C must have at least `size` elements")

    # Match the CUDA launch configuration: blockSize = 960
    BLOCK = 960
    # Choose a CHUNK (vector width) that equals num_warps * 32 used in the @triton.jit decorator.
    # Here num_warps=8 => CHUNK = 8 * 32 = 256
    CHUNK = 256

    num_blocks = (n + BLOCK - 1) // BLOCK
    if num_blocks == 0:
        return  # nothing to do

    grid = (num_blocks,)

    # Launch Triton kernel. The BLOCK and CHUNK are passed as compile-time constants.
    _triton_kernel_impl[grid](A, B, C, n, BLOCK=BLOCK, CHUNK=CHUNK)