import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: element‑wise addition
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # float* __restrict__ A
    B_ptr,          # float* __restrict__ B
    C_ptr,          # float* __restrict__ T_add (output)
    N,              # number of elements to process (runtime)
    BLOCK_SIZE: tl.constexpr,  # compile‑time block size (must be power of 2)
):
    """
    Compute C[i] = A[i] + B[i] for i in [0, N).

    The kernel processes BLOCK_SIZE elements per program (i.e. per block).
    """
    pid = tl.program_id(0)                     # block index
    offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper that mimics the original CUDA launch configuration
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton entry point equivalent to the original CUDA kernel.

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors residing on the same CUDA device.
    size : int
        Number of elements to process (must not exceed the length of the tensors).
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must have dtype torch.float32.")
    if A.dim() != 1 or B.dim() != 1 or C.dim() != 1:
        raise RuntimeError("All tensors must be 1‑D.")
    if size < 0:
        raise ValueError("size must be non‑negative.")

    # Ensure the requested size fits within the tensors
    max_len = min(A.shape[0], B.shape[0], C.shape[0])
    if size > max_len:
        raise ValueError(f"size ({size}) exceeds the smallest tensor length ({max_len}).")
    if size == 0:
        return  # nothing to do

    # ------------------------------------------------------------------
    # Triton launch configuration
    # ------------------------------------------------------------------
    # Triton requires the range of tl.arange to be a power of two.
    # 256 is the largest power‑of‑two ≤ 320 and works well on H800.
    BLOCK_SIZE = 256

    # Number of program instances (grid size)
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A, B, C, size, BLOCK_SIZE=BLOCK_SIZE
    )
    # Synchronize to make the kernel completion observable (useful for testing)
    torch.cuda.synchronize()