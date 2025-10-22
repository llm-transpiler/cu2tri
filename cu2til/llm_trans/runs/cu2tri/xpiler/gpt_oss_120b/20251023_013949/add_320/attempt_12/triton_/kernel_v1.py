import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing element‑wise addition.
# Named exactly as required: `_triton_kernel_impl`.
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # float* __restrict__ A
    B_ptr,          # float* __restrict__ B
    C_ptr,          # float* __restrict__ T_add (output)
    size,           # int: number of elements to process
    BLOCK_SIZE: tl.constexpr,  # compile‑time constant (320)
):
    # Program (block) identifier
    pid = tl.program_id(0)

    # Compute the global offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Guard against out‑of‑bounds accesses
    mask = offsets < size

    # Load inputs (masked)
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    # Store the result (masked)
    tl.store(C_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function matching the original CUDA entry point.
# Signature: (float* A, float* B, float* C, int size)
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton wrapper that mirrors the original CUDA kernel's behavior.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (>= size,) on CUDA device, dtype torch.float32.
    B : torch.Tensor
        Input tensor of shape (>= size,) on CUDA device, dtype torch.float32.
    C : torch.Tensor
        Output tensor of shape (>= size,) on CUDA device, dtype torch.float32.
    size : int
        Number of elements to process.
    """
    # ------------------------------------------------------------------
    # Input validation (mirrors typical CUDA expectations)
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must reside on a CUDA device.")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("All tensors must be of type torch.float32.")
    if not (A.numel() >= size and B.numel() >= size and C.numel() >= size):
        raise RuntimeError("Tensor storage must be at least 'size' elements long.")
    if size < 0:
        raise ValueError("size must be non‑negative.")

    # ------------------------------------------------------------------
    # Kernel launch configuration
    # ------------------------------------------------------------------
    BLOCK_SIZE = 320  # matches __launch_bounds__(320) in the CUDA code
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
    )