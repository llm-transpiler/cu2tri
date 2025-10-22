import torch
import triton
import triton.language as tl

# Block size matches the original CUDA block size of 1024 threads
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton implementation of the element‑wise addition kernel.
    The kernel mirrors the original CUDA kernel which only processes
    indices < 2304, regardless of the `size` argument.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices for this block

    # Hard‑coded bound from the original CUDA kernel
    mask = offsets < 2304

    # Load with mask to avoid out‑of‑bounds memory accesses
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    # Compute and store only for valid indices
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point wrapper that mimics the original CUDA kernel launch.
    Parameters
    ----------
    A, B, C : torch.Tensor
        Input and output tensors. Must be CUDA tensors of dtype torch.float32.
    size : int
        Logical size of the vectors (mirrors the original `size` argument).
        The kernel still respects the hard‑coded bound of 2304 elements.
    """
    # Basic sanity checks
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("Only torch.float32 tensors are supported")
    # Ensure contiguous memory layout
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # Compute grid size exactly as in the CUDA launch
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel.
    # `num_warps=32` yields 32*32 = 1024 threads per block, matching the original launch bounds.
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32
    )
    # Optional: synchronize to make kernel completion explicit    torch.synchronize()