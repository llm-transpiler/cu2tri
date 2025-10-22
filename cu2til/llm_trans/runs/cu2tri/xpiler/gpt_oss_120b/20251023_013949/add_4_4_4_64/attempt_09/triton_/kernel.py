import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: element‑wise addition with the same semantics as the CUDA kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Parameters
    ----------
    A_ptr, B_ptr, C_ptr : pointers to float32 tensors (passed as torch tensors)
    BLOCK_SIZE          : compile‑time constant (1024)
    """
    pid = tl.program_id(0)  # block index (equivalent to blockIdx.x)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global linear indices
    mask = offsets < 4096  # hard‑coded bound from the original CUDA kernel

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)


# ----------------------------------------------------------------------
# Wrapper matching the original CUDA host function signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Mimics the original `cuda_kernel` entry point.

    Arguments
    ---------
    A, B, C : 1‑D torch.float32 tensors on the CUDA device
    size    : logical vector length (used only for grid computation)
    """
    # ---- sanity checks (identical to expectations of the original CUDA wrapper) ----
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on a CUDA device"
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
    grid = (num_blocks,)  # Triton expects a tuple for the grid dimensions

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE=BLOCK_SIZE)

    # Ensure kernel completion before returning
    torch.cuda.synchronize()