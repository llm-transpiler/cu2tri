import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: mirrors the original CUDA kernel behavior
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr: tl.pointer(tl.float32),
    B_ptr: tl.pointer(tl.float32),
    C_ptr: tl.pointer(tl.float32),
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element‑wise addition kernel.
    - 1024 threads per block (BLOCK_SIZE)
    - Guard against global index >= 4096 (hard‑coded bound from the CUDA code)
    """
    pid = tl.program_id(0)  # block index (equivalent to blockIdx.x)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global linear indices
    mask = offsets < 4096  # same bound used in the original kernel

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)


# ----------------------------------------------------------------------
# Wrapper that matches the original CUDA host function signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that mimics the original ``cuda_kernel`` signature.

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors residing on the CUDA device.
    size : int
        Logical size of the vectors (used only to compute the grid size,
        exactly as the CUDA host code does).
    """
    # ------------------------------------------------------------------
    # Sanity checks (identical to expectations of the original CUDA wrapper)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    BLOCK_SIZE = 1024
    # Compute grid size exactly like the CUDA host code:
    #   dim3 numBlocks((size + 1024 - 1) / 1024);
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)  # Triton expects a tuple for the grid

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE=BLOCK_SIZE)

    # Synchronize to make the kernel effect visible before returning
    torch.cuda.synchronize()