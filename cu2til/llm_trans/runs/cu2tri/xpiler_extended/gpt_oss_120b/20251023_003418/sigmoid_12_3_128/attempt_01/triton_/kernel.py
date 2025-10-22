import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, compute_ptr, BLOCK_SIZE: tl.constexpr):
    # Program ID corresponds to CUDA's blockIdx.x
    pid = tl.program_id(0)
    # Offsets for each thread within the block (equivalent to threadIdx.x)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Guard against the hard‑coded bound used in the original CUDA kernel
    mask = offs < 4608
    # Load input values (masked loads return 0.0 for out‑of‑range lanes)
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    # Compute sigmoid: 1 / (1 + exp(-a))
    result = 1.0 / (1.0 + tl.exp(-a))
    # Store the result (masked store)
    tl.store(compute_ptr + offs, result, mask=mask)

def triton_kernel(A: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton equivalent of the original CUDA kernel.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA device.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32 on CUDA device.
    size : int
        Number of elements to process (mirrors the original `size` argument).
    """
    # Basic sanity checks (mirrors expectations of the CUDA version)
    assert A.is_cuda and C.is_cuda, "Tensors must reside on a CUDA device"
    assert A.dtype == torch.float32 and C.dtype == torch.float32, "Only float32 supported"
    # Ensure contiguous memory layout
    A = A.contiguous()
    C = C.contiguous()
    BLOCK_SIZE = 1024
    # Compute grid size exactly as the CUDA launch configuration
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, C, BLOCK_SIZE=BLOCK_SIZE)