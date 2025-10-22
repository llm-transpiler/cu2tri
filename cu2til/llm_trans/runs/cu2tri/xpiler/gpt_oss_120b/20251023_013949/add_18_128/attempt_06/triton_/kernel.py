import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK_SIZE: tl.constexpr):
    # Program (block) ID
    pid = tl.program_id(0)
    # Thread (lane) offsets within the block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Hard‑coded bound identical to the CUDA implementation
    mask = offsets < 2304
    # Load with mask (out‑of‑range loads return 0.0)
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    # Store the result
    tl.store(T_add_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton equivalent of the original CUDA kernel.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32.
    size : int
        Logical size of the vectors (used only for grid calculation).
    """
    # Validate inputs
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "Tensors must be contiguous"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "Only float32 supported"
    # Block size matches the CUDA launch bounds
    BLOCK_SIZE = 1024
    # Compute grid size exactly as the CUDA host code
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE
    )