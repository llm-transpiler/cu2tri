import triton
import triton.language as tl
import torch


# Triton kernel implementation must be named exactly `_triton_kernel_impl`.
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    rng = tl.arange(0, CHUNK)

    # Process the block in CHUNK-sized lanes (CHUNK is a compile-time constant)
    for i in range(0, BLOCK, CHUNK):
        offs = block_start + i + rng
        # mask out lanes that are outside the array bounds or outside the BLOCK
        mask = (offs < size) & ((i + rng) < BLOCK)
        a = tl.load(A_ptr + offs, mask=mask, other=0.0)
        b = tl.load(B_ptr + offs, mask=mask, other=0.0)
        c = a + b
        tl.store(C_ptr + offs, c, mask=mask)


# Wrapper matching the CUDA kernel signature exactly:
# triton_kernel(float *A, float *B, float *C, int size)
def triton_kernel(A, B, C, size):
    """
    A, B, C: torch.Tensor on CUDA, dtype=torch.float32.
    size: number of elements to process (int)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Triton kernels require a CUDA device.")

    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor) or not isinstance(C, torch.Tensor):
        raise TypeError("A, B, and C must be torch.Tensor")

    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("A, B, and C must be torch.float32")

    if not A.is_cuda or not B.is_cuda or not C.is_cuda:
        raise ValueError("A, B, and C must be on CUDA device")

    # Helper: ensure a 1-D contiguous buffer for Triton. If the input is already 1-D & contiguous,
    # return it as-is (no copy). Otherwise, create a cloned contiguous 1-D tensor (independent memory).
    def _ensure_1d_contig_clone(x):
        if x.dim() == 1 and x.is_contiguous():
            return x, False
        # clone() ensures a new buffer (no aliasing) and contiguous memory; view(-1) flattens to 1-D
        return x.clone().contiguous().view(-1), True

    orig_A, orig_B, orig_C = A, B, C
    A_, a_copied = _ensure_1d_contig_clone(orig_A)
    B_, b_copied = _ensure_1d_contig_clone(orig_B)
    C_, c_copied = _ensure_1d_contig_clone(orig_C)

    n = int(size)

    if A_.numel() < n or B_.numel() < n or C_.numel() < n:
        raise ValueError("Buffers A, B, and C must have at least `size` elements")

    # Match the CUDA launch configuration: blockSize = 960
    BLOCK = 960
    # CHUNK is the number of lanes (threads per program); choose a reasonable compile-time value.
    # Using 256 (8 warps * 32) is fine for many GPUs; it's a compile-time constant passed to the kernel.
    CHUNK = 256

    num_blocks = (n + BLOCK - 1) // BLOCK
    if num_blocks == 0:
        return

    grid = (num_blocks,)

    # Launch Triton kernel. Pass num_warps as a launch-time kwarg.
    _triton_kernel_impl[grid](A_, B_, C_, n, BLOCK=BLOCK, CHUNK=CHUNK, num_warps=8)

    # If we created a cloned contiguous buffer for C, copy results back into the original C.
    if c_copied:
        # reshape the contiguous 1-D result back to the original shape of C and copy.
        # Using clone() earlier guarantees no aliasing between src (C_) and dst (orig_C).
        orig_C.copy_(C_.view(orig_C.shape))