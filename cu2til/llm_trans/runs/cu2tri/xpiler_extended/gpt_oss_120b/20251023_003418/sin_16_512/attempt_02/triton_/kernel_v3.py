import torch
import triton
import triton.language as tl

# Threads per block (matches the original CUDA kernel)
BLOCK_SIZE = 256

@triton.jit
def _triton_kernel_impl(input_ptr, output_ptr, total, BLOCK_SIZE: tl.constexpr):
    """Triton kernel that computes sin(input) element‑wise."""
    pid = tl.program_id(0)  # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # linear indices for this block
    mask = offsets < total  # guard against out‑of‑bounds

    # Load, compute, and store (masked)
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.sin(x)
    tl.store(output_ptr + offsets, y, mask=mask)

def triton_kernel(input: torch.Tensor, output: torch.Tensor, total: int):
    """
    Triton entry point mirroring the original CUDA kernel signature.

    Parameters
    ----------
    input : torch.Tensor
        CUDA tensor of dtype torch.float32 (any shape).
    output : torch.Tensor
        CUDA tensor of dtype torch.float32 (same shape as ``input`` or larger).
    total : int
        Number of elements to process.
    """
    # ---- Input validation -------------------------------------------------
    if not (input.is_cuda and output.is_cuda):
        raise RuntimeError("Both input and output must be CUDA tensors.")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("Both input and output must be of type torch.float32.")
    if total < 0:
        raise ValueError("`total` must be non‑negative.")
    if total > input.numel():
        raise RuntimeError("`total` exceeds the number of elements in the input tensor.")
    if total > output.numel():
        raise RuntimeError("`total` the number of elements in the output tensor    # ---- Prepare contiguous ‑D views for Triton -------------------------
    input_flat = input.reshape(-1).contiguous()
    output_flat = output.reshape(-1).contiguous()

    # ---- Launch configuration ---------------------------------------------
    grid = (triton.cdiv(total, BLOCK_SIZE),)  # one program per BLOCK_SIZE elements

    # ---- Kernel launch ----------------------------------------------------
    _triton_kernel_impl[grid](
        input_flat,
        output_flat,
        total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.cuda.synchronize()