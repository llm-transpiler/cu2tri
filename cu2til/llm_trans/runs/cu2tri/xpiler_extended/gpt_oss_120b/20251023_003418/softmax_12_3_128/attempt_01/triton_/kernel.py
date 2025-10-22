import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, C_ptr, size1, size2, BLOCK_SIZE: tl.constexpr):
    """Row‑wise softmax kernel.

    Each program (block) processes one row of length `size2` (expected 128).
    """
    pid = tl.program_id(0)               # row index
    row = pid
    if row >= size1:
        return

    # column indices within the row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < size2                    # handle size2 != BLOCK_SIZE safely

    # linear offset for A and C
    offset = row * size2 + cols

    # Load A; masked out‑of‑range columns get -inf so they don't affect the max
    a = tl.load(A_ptr + offset, mask=mask, other=-float('inf'))

    # Compute max across the row
    max_val = tl.max(a, axis=0)

    # Exponentiate shifted values
    exp_val = tl.exp(a - max_val)

    # Sum of exponentials (denominator)
    denom = tl.sum(exp_val, axis=0)

    # Softmax output
    soft = exp_val / denom

    # Store result
    tl.store(C_ptr + offset, soft, mask=mask)


def triton_kernel(A, C, size1, size2):
    """
    Compute row‑wise softmax of `A` and store the result in `C`.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size1, size2), dtype torch.float32, on CUDA.
    C : torch.Tensor
        Output tensor of the same shape and dtype as `A`, on CUDA.
    size1 : int
        Number of rows.
    size2 : int
        Number of columns (expected 128).
    """
    import torch

    # ----------------------------------------------------------------------
    # Input validation
    # ----------------------------------------------------------------------
    if not (isinstance(A, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A and C must be torch.Tensor")
    if not (A.is_cuda and C.is_cuda):
        raise RuntimeError("A and C must be CUDA tensors")
    if A.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("A and C must be float32 tensors")
    if A.shape != (size1, size2) or C.shape != (size1, size2):
        raise RuntimeError(f"Tensor shapes must be ({size1}, {size2})")

    # Ensure contiguous layout for optimal memory access
    A = A.contiguous()
    C = C.contiguous()

    # ----------------------------------------------------------------------
    # Kernel launch configuration
    # ----------------------------------------------------------------------
    BLOCK_SIZE = 128               # compile‑time constant: columns per row
    grid = (size1,)                # one program per row

    # Launch Triton kernel
    _triton_kernel_impl[grid](
        A,
        C,
        size1,
        size2,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,               # 128 threads = 4 warps
    )