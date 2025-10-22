import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Compile‑time constant: number of threads per block (matches CUDA launch)
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024

# ----------------------------------------------------------------------
# Triton kernel: element‑wise addition C = A + B
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # float* __restrict__ A
    B_ptr,               # float* __restrict__ B
    C_ptr,               # float* __restrict__ C (output)
    size,                # total number of elements (runtime)
    BLOCK_SIZE: tl.constexpr,  # compile‑time block size
):
    pid = tl.program_id(0)                     # block index
    block_start = pid * BLOCK_SIZE              # first element this block handles
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # indices for this block

    mask = offsets < size                       # guard against out‑of‑bounds
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper that mirrors the original CUDA API
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton entry point reproducing the behavior of the original CUDA kernel.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA) of shape (size,).
    B : torch.Tensor
        Input tensor (float32, CUDA) of shape (size,).
    C : torch.Tensor
        Output tensor (float32, CUDA) of shape (size,).
    size : int
        Number of elements to process.
    """
    # ------------------------------------------------------------------
    # Input validation (same expectations as the CUDA version)
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32")
    if A.shape != B.shape or A.shape != C.shape:
        raise RuntimeError("All tensors must have the same shape")

    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # ------------------------------------------------------------------
    # Grid configuration (identical to the CUDA launch)
    # ------------------------------------------------------------------
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # 1‑D grid

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    # Pass BLOCK_SIZE as a compile‑time constant; num_warps=32 yields 1024 threads per block
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE, num_warps=32)

    # Synchronize to make the kernel completion visible to the host
    torch.cuda.synchronize()