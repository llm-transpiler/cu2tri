import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Triton kernel: elementwise addition A + B -> C
# -------------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *float32
    B_ptr,          # *float32
    C_ptr,          # *float32
    size: tl.int32, # total number of elements to process
    BLOCK_SIZE: tl.constexpr,  # compile‑time constant: threads per block
):
    """
    Each program (block) processes BLOCK_SIZE elements.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # absolute element indices
    mask = offsets < size                       # out‑of‑bounds mask

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


# -------------------------------------------------------------------------
# Wrapper matching the original CUDA kernel signature
# -------------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point mirroring the original CUDA kernel signature.
    Parameters
    ----------
    A, B, C : torch.Tensor
        CUDA tensors of dtype torch.float32. ``C`` is the output tensor.
    size : int
        Number of elements to process (must not exceed the length of the tensors).
    """
    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"

    max_len = min(A.numel(), B.numel(), C.numel())
    assert size <= max_len, f"Requested size ({size}) exceeds tensor length ({max_len})"

    # -----------------------------------------------------------------
    # Triton launch configuration
    # -----------------------------------------------------------------
    BLOCK_SIZE = 1024  # matches __launch_bounds__(1024) in the CUDA code
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # 1‑D grid

    # -----------------------------------------------------------------
    # Kernel launch
    # -----------------------------------------------------------------
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

# -------------------------------------------------------------------------
# Example usage (uncomment for a quick sanity check)
# -------------------------------------------------------------------------
# if __name__ == "__main__":
#     size = 2048000
#     A = torch.randn(size, device="cuda", dtype=torch.float32)
#     B = torch.randn(size, device="cuda", dtype=torch.float32)
#     C = torch.empty_like(A)
#     triton_kernel(A, B, C, size)
#     torch.testing.assert_allclose(C, A + B)