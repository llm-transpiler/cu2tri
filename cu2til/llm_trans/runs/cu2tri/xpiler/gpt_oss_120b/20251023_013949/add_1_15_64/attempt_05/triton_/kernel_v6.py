import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementation
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *Pointer* to input tensor A (float32)
    B_ptr,          # *Pointer* to input tensor B (float32)
    T_add_ptr,      # *Pointer* to output tensor C (float32)
    size,           # Number of elements to process (runtime)
    BLOCK_SIZE: tl.constexpr,  # Compile‑time constant: block size (must be power of 2)
):
    """
    Element‑wise addition: T_add[i] = A[i] + B[i] for i in [0, size).
    Each program instance processes BLOCK_SIZE consecutive elements.
    """
    pid = tl.program_id(0)                                 # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices
    mask = offsets < size                                   # out‑of‑bounds guard

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(T_add_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Launches the Triton kernel to compute C = A + B for `size` elements.

    Parameters
    ----------
    A, B, C : torch.Tensor
        CUDA tensors of dtype torch.float32. Must contain at least `size` elements.
    size : int
        Total number of elements to process (linear length).
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("All tensors must have dtype torch.float32")

    # Flatten tensors to 1‑D for linear indexing (preserves contiguity)
    A_flat = A.view(-1)
    B_flat = B.view(-1)
    C_flat = C.view(-1)

    # Determine the effective number of elements to process
    effective_size = min(int(size), A_flat.numel(), B_flat.numel(), C_flat.numel())
    if effective_size <= 0:
        return  # nothing to do

    # Choose a power‑of‑2 block size (tl.arange requires it)
    BLOCK_SIZE = 1024  # 2^10, >= 960 and satisfies the power‑of‑2 constraint

    # Compute grid dimensions: one block per BLOCK_SIZE elements
    grid = ((effective_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A_flat,
        B_flat,
        C_flat,
        effective_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Optional synchronization for debugging
    # torch.cuda.synchronize()