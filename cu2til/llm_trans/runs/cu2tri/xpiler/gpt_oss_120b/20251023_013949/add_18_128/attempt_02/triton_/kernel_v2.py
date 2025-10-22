import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Triton kernel: addition with a‑coded bound of 2304 elements.
# -------------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A, B, T_add, BLOCK_SIZE: tl.constexpr):
    # program id == blockIdx.x
    pid = tl.program_id(0)
    # global offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # mask enforces the original CUDA bound (2304)
    mask = offsets < 2304

    # load with masked loads (out‑of‑range elements yield 0.0)
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)

    # store the sum with a masked store
    tl.store(T_add + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point mirroring the original CUDA wrapper:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D tensors of dtype torch.float32 on the CUDA device.
    size : int
        Logical vector length (used only to compute launch configuration).
    """
    # -----------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    assert A.dim() == B.dim() == C.dim() == 1, "All tensors must be 1‑D"

    # -----------------------------------------------------------------
    # Launch configuration (matches __launch_bounds__(1024) in CUDA)
    # -----------------------------------------------------------------
    BLOCK_SIZE = 1024
    # Number of blocks needed to cover `size` elements (rounded up)
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # -----------------------------------------------------------------
    # Kernel launch
    # -----------------------------------------------------------------
    _triton_kernel_impl[(num_blocks,)](A, B, C, BLOCK_SIZE=BLOCK_SIZE)