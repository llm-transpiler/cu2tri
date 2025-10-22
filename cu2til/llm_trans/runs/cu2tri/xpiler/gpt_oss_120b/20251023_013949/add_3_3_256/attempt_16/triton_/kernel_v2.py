import torch
import triton
import triton.language as tl

# Compile-time constants matching the original CUDA launch configuration
BLOCK_SIZE = 1024          # threads per block (launch_bounds)
MAX_ELEMS = 2304           # hard‑coded bound in the original kernel

@triton.jit
def _triton_kernel_impl(
    A_ptr, B_ptr, C_ptr,
    size,                     # kept for signature compatibility; unused
    BLOCK_SIZE: tl.constexpr,
    MAX_ELEMS: tl.constexpr
):
    """
    Triton implementation of the element‑wise addition kernel.
    Mirrors the behavior of the original CUDA kernel:
      - Each thread processes one element.
      - Only indices < MAX_ELEMS (2304) are processed.
    """
    pid = tl.program_id(0)                                 # block index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)     # global indices
    mask = offs < MAX_ELEMS                                 # replicate CUDA if‑guard

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(C_ptr + offs, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original `cuda_kernel` signature.
    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D contiguous CUDA tensors of dtype torch.float32.
    size : int
        Logical size of the vectors (used only to compute the grid size,
        exactly as the original CUDA launch configuration).
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"

    # Compute grid dimensions exactly like the CUDA wrapper
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_ELEMS=MAX_ELEMS,
        num_warps=32  # 1024 threads = 32 warps
    )
    # The original CUDA launch is asynchronous; we keep the same semantics.
    # Uncomment the line below if a synchronous launch is required.
    # torch.cuda.synchronize()