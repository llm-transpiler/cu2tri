import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Configuration (mirrors the original CUDA launch bounds)
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024  # Threads per block, matches __launch_bounds__(1024)

# ----------------------------------------------------------------------
# Triton kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    Compute C = A + B for the first N elements.

    Parameters
    ----------
    A_ptr, B_ptr, C_ptr : pointers to float32 tensors (device memory)
    N : number of valid elements (equivalent to `size` in the CUDA wrapper)
    BLOCK_SIZE : compile‑time constant, threads per block
    """
    pid = tl.program_id(0)  # Block index (1‑D grid)
    # Absolute indices for the threads in this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask out-of-range threads
    mask = offsets < N

    # Load values with masking; out‑of‑range loads return 0.0 (won't affect result)
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    # Element‑wise addition
    c = a + b

    # Store the result, respecting the mask
    tl.store(C_ptr + offsets, c, mask=mask)

# ----------------------------------------------------------------------
# Wrapper function (identical signature to the original CUDA entry point)
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton wrapper replicating the behavior of the original CUDA kernel.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32) on CUDA device.
    B : torch.Tensor
        Input tensor (float32) on CUDA device.
    C : torch.Tensor
        Output tensor (float32) on CUDA device.
    size : int
        Number of elements to process (matches the `size` argument in the CUDA wrapper).
    """
    # ------------------------------------------------------------------
    # Sanity checks (mirroring typical CUDA expectations)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor storage must be at least `size` elements"

    # ------------------------------------------------------------------
    # Compute grid dimensions based on the provided size (identical to CUDA launch)
    # ------------------------------------------------------------------
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # 1‑D grid

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

    # ------------------------------------------------------------------
    # Ensure kernel completion before returning
    # ------------------------------------------------------------------
    torch.cuda.synchronize()