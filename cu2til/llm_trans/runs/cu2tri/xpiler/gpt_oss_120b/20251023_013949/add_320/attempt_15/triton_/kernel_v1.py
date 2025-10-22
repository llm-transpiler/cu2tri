import torch
import triton
import triton.language as tl

# Triton kernel that mirrors the original CUDA kernel behavior.
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, BLOCK_SIZE: tl.constexpr):
    # Offsets correspond to threadIdx.x in the CUDA kernel.
    offsets = tl.arange(0, BLOCK_SIZE)
    a = tl.load(A_ptr + offsets)
    b = tl.load(B_ptr + offsets)
    tl.store(T_add_ptr + offsets, a + b)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that replicates the original cuda_kernel launch.
    Parameters:
        A (torch.Tensor): Input tensor (float32, CUDA).
        B (torch.Tensor): Input tensor (float32, CUDA).
        C (torch.Tensor): Output tensor (float32, CUDA).
        size (int): Number of elements (used only for grid sizing).
    """
    # Validate inputs.
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be on a CUDA device.")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise RuntimeError("All tensors must have dtype torch.float32.")
    
    BLOCK_SIZE = 320  # Matches __launch_bounds__(320) in the CUDA code.
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel. BLOCK_SIZE is a compile‑time constant.
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE=BLOCK_SIZE)