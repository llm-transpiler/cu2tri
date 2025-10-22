import torch
import triton
import triton.language as tl

# Triton kernel implementation (must be named exactly as requested)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, bound, BLOCK: tl.constexpr):
    """
    A_ptr, B_ptr, C_ptr - device pointers (provided by passing torch tensors)
    bound - int: the upper-exclusive index bound to guard the operation (mirrors the CUDA kernel's 2304)
    BLOCK - number of elements handled per program instance (tl.constexpr)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < bound
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(C_ptr + offs, a + b, mask=mask)


# Wrapper entry-point (must be named exactly as requested and keep same signature semantics)
def triton_kernel(A, B, C, size):
    """
    Entry point that mirrors the original CUDA kernel launcher:
      - A, B, C : torch.Tensor on CUDA (any shape); they will be treated as flattened buffers
      - size    : int (used for grid computation, as in the original CUDA launcher)
    Behavior is identical to the original CUDA code:
      - block size = 1024
      - numBlocks = (size + 1024 - 1) // 1024
      - the device kernel writes C[idx] = A[idx] + B[idx] only for idx < 2304 (constant),
        matching the original kernel's conditional.
    """
    # Basic validations
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor on CUDA device")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must be float32 tensors")

    # Flatten the tensors (try to avoid copies when possible)
    # reshape will return a view when possible; if it's not contiguous, make a contiguous copy.
    A_flat = A.reshape(-1)
    if not A_flat.is_contiguous():
        A_flat = A_flat.contiguous()

    B_flat = B.reshape(-1)
    if not B_flat.is_contiguous():
        B_flat = B_flat.contiguous()

    C_flat_view = C.reshape(-1)
    needs_copy_back = False
    if not C_flat_view.is_contiguous():
        # Make a contiguous buffer for the kernel to write into, and mark to copy back later.
        C_flat = C_flat_view.contiguous()
        needs_copy_back = True
    else:
        C_flat = C_flat_view

    # Ensure size is an int for grid calculation
    size = int(size)

    BLOCK = 1024
    num_blocks = (size + BLOCK - 1) // BLOCK if size > 0 else 0
    if num_blocks == 0:
        return

    grid = (num_blocks,)

    # The original CUDA kernel uses a hard-coded bound of 2304 for the conditional.
    # To be safe against short input buffers, we also clamp to the available buffer sizes.
    # This preserves the semantic "only indices < 2304 are written" while avoiding OOB in Triton.
    bound = 2304
    # clamp bound to available elements to avoid invalid memory accesses
    available = min(A_flat.numel(), B_flat.numel(), C_flat.numel())
    bound = min(bound, int(available))

    # Launch Triton kernel. Pass `bound` to mirror the original conditional (2304).
    _triton_kernel_impl[grid](A_flat, B_flat, C_flat, bound, BLOCK=BLOCK)

    # If we had to allocate a contiguous output buffer, copy results back to the original C layout.
    if needs_copy_back:
        # reshape contiguous flat result back to original shape and copy into original tensor
        C.copy_(C_flat.reshape(C.shape))