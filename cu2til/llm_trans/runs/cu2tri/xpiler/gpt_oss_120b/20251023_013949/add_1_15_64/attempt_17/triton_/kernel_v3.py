import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
# The original CUDA kernel used a block size of 960 threads.
BLOCK_SIZE = 960  # must be a compile‑time constant for Triton

# ----------------------------------------------------------------------
# Triton kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # pointer to input tensor A
    B_ptr,          # pointer to input tensor B
    T_add_ptr,      # pointer to output tensor T_add (C in the wrapper)
    N,              # total number of elements to process
    BLOCK_SIZE: tl.constexpr,  # compile‑time block size (960)
):
    """
    Element‑wise addition kernel.
    Mirrors the intended functionality of the original CUDA kernel:
        T_add[i] = A[i] + B[i]  for i in [0, N)
    The kernel processes BLOCK_SIZE elements per program instance (block).
    """
    pid = tl.program_id(0)                     # block index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global offsets
    mask = offs < N                             # guard for the last (partial) block

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
 tl.store(T_add_ptr + offs, a + b, mask=mask# ----------------------------------------------------------------# Wrapper function (entry point)
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Launches the Triton kernel with the same launch configuration as the
    original CUDA code.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA).
    B : torch.Tensor
        Input tensor B (float32, CUDA).
    C : torch.Tensor
        Output tensor where the sum is written (float32, CUDA).
    size : int
        Number of elements to process.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError("All tensors must be of type torch.float32")
    if A.shape != B.shape or A.shape != C.shape:
        raise ValueError("All tensors must have the same shape")
    if size != A.numel():
        # Allow size to be less than the tensor length (partial work),
        # but it must not exceed the allocation.
        if size > A.numel():
            raise ValueError("size exceeds tensor length")
    # ------------------------------------------------------------------
    # Grid configuration (mirrors the original CUDA launch)
    # ------------------------------------------------------------------
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    if num_blocks == 0:
        return  # nothing to do

    grid = (num_blocks,)

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    # Triton requires the number of warps to be a power of two.
    # 32 warps = 1024 threads per block, which comfortably covers BLOCK_SIZE=960.
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )