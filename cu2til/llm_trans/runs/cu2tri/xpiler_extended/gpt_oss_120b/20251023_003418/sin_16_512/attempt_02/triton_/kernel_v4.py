import torch
import triton
import triton.language as tl

# Number of threads per block (matches the original CUDA kernel)
BLOCK_SIZE = 256

@triton.jit
def _triton_kernel_impl(input_ptr, output_ptr, total, BLOCK_SIZE: tl.constexpr):
    """Elementwise sine kernel.

    Computes `output[i] = sin(input[i])` for `i < total`.
    """
    pid = tl.program_id(0)  # block (program) index
    # Linear indices for this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    # Load, compute, and store (masked)
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.sin(x)
    tl.store(output_ptr + offsets, y, mask=mask)

def triton_kernel(input: torch.Tensor, output: torch.Tensor, total: int):
    """Entry point mirroring the original CUDA kernel signature.

    Parameters
    ----------
    input : torch.Tensor
        CUDA tensor of dtype torch.float32.
    output : torch.Tensor
        CUDA tensor of dtype torch.float32.
    total : int
        Number of elements to process.
    """
    # Convert possible torch scalar to Python int
    total = int(total)

    # ---- Validation -------------------------------------------------------
    if not (input.is_cuda and output.is_cuda):
        raise RuntimeError("Both input and output must be CUDA tensors.")
    if input.device != output.device:
        raise RuntimeError("Input and output must be on the same CUDA device.")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("Both input and output must have dtype torch.float32.")
    if total < 0:
        raise ValueError("`total` must be non‑negative.")
    if total == 0:
        # Nothing to do
        return
    if total > input.numel():
        raise RuntimeError("`total` exceeds the number of elements in the input tensor.")
    if total > output.numel():
        raise RuntimeError("`total` exceeds the number of elements in the output tensor.")

    # ---- Prepare contiguous 1‑D views --------------------------------------
    input_flat = input.view(-1).contiguous()
    output_flat = output.view(-1).contiguous()

    # ---- Launch configuration ---------------------------------------------
    grid = (triton.cdiv(total, BLOCK_SIZE),)  # one program per BLOCK_SIZE elements

    # ---- Kernel launch -----------------------------------------------------
    _triton_kernel_impl[grid](
        input_flat,
        output_flat,
        total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.cuda.synchronize()

__all__ = ["triton_kernel"]