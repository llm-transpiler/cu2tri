import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK_SIZE: tl.constexpr):
    # Program (block) ID
    pid = tl.program_id(0)
    # Compute linear offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Hard‑coded bound from the original CUDA kernel
    mask = offsets < 4032
    # Load inputs with mask (out‑of‑range loads return 0.0, which will be ignored)
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    # Perform element‑wise addition
    out = a + b
    # Store result with mask
    tl.store(T_add_ptr + offsets, out, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton equivalent of the original CUDA kernel.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32) on CUDA.
    B : torch.Tensor
        Input tensor (float32) on CUDA.
    C : torch.Tensor
        Output tensor (float32) on CUDA.
    size : int
        Logical size of the vectors (used only for grid sizing, mirroring the CUDA launch).
    """
    # Sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    # Ensure contiguous memory layout
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    BLOCK_SIZE = 1024
    # Compute grid size exactly as in the CUDA wrapper
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)