import torch
import triton
import triton.language as tl

# Triton kernel implementation
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Element‑wise addition kernel.
    Mirrors the original CUDA kernel that processes up to 2304 elements.
    """
    pid = tl.program_id(0)                     # block index
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start

    # Original CUDA kernel only processes indices < 2304.
    # We also guard against out‑of‑bounds accesses using `size`.
    mask = (offsets < size) & (offsets < 2304)

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA) of length at least `size`.
    B : torch.Tensor
        Input tensor (float32, CUDA) of length at least `size`.
    C : torch.Tensor
        Output tensor (float32, CUDA) of length at least `size`.
    size : int
        Number of elements to process (the kernel will still cap at 2304).
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    # Ensure contiguous memory layout
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    BLOCK_SIZE = 1024  # matches __launch_bounds__(1024) in CUDA

    # Compute grid size: number of program instances (blocks)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)