import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise subtraction
@triton.jit
def _triton_kernel_impl(
    a_ptr,          # const float* __restrict__ a
    b_ptr,          # const float* __restrict__ b
    out_ptr,        # float* __restrict__ output
    total,          # int total
    BLOCK_SIZE: tl.constexpr  # compile‑time block size
):
    pid = tl.program_id(0)                     # 1‑D grid
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = a - b
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_kernel(a: torch.Tensor,
                  b: torch.Tensor,
                  output: torch.Tensor,
                  total: int):
    """
    Wrapper that mimics the original CUDA entry point:
        void cuda_kernel(const float* a, const float* b, float* output, int total)

    Parameters
    ----------
    a : torch.Tensor
        Input tensor ``a`` (float32, CUDA, contiguous).
    b : torch.Tensor
        Input tensor ``b`` (float32, CUDA, contiguous).
    output : torch.Tensor
        Output tensor (float32, CUDA, contiguous) where ``output[i] = a[i] - b[i]``.
    total : int
        Number of elements to process.
    """
    # Basic validation – mirrors the expectations of the CUDA version
    if not (a.is_cuda and b.is_cuda and output.is_cuda):
        raise RuntimeError("All tensors must reside on the CUDA device.")
    if a.dtype != torch.float32 or b.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32.")
    if not (a.is_contiguous() and b.is_contiguous() and output.is_contiguous()):
        raise RuntimeError("All tensors must be contiguous.")
    if total > a.numel() or total > b.numel() or total > output.numel():
        raise ValueError("`total` exceeds the size of one or more tensors.")

    # Choose a block size that works well on modern GPUs (e.g., 256 matches the original CUDA launch)
    BLOCK_SIZE = 256

    # Compute grid dimensions
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        a,
        b,
        output,
        total,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4  # typical for FP32 arithmetic on NVIDIA GPUs
    )