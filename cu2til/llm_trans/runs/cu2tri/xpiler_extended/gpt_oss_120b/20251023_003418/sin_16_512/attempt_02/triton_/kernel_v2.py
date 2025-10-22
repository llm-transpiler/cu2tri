import torch
import triton
import triton.language as tl

# Block size matches the CUDA version's threads per block
BLOCK_SIZE = 256

@triton.jit
def _triton_kernel_impl(input_ptr, output_ptr, total, BLOCK_SIZE: tl.constexpr):
    """Triton kernel that computes sin of input elements."""
    pid = tl.program_id(0)  # program (block) index
    offsets = pid * + tl.arange(0, BLOCK_SIZE)  # linear offsets
    mask = offsets < total

    # Load input values (masked)
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Compute sine
    y = tl.sin(x)
    # Store results (masked)
    tl.store(output_ptr + offsets, y, mask=mask)

def triton_kernel(input: torch.Tensor, output: torch.Tensor, total: int):
    """
    Triton entry point mirroring the original CUDA kernel signature.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor (any shape) on CUDA device with dtype torch.float32.
    output : torch.Tensor
        Output tensor ( shape) on CUDA device with dtype.float32.
    total : int
        Number of elements to process.
    """
    # Validate and normalize `total`
    if not isinstance(total, (int, torch.Tensor)):
        raise TypeError("`total` must be an integer or a scalar tensor.")
    total = int(total)
    if total < 0:
        raise ValueError("`total` must be non‑negative.")
    if total == 0:
        return

    # Basic tensor checks
    if not input.is_cuda or not output.is_cuda:
        raise RuntimeError("Both input and output must be tensors.")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("Both and output must be of type torch.float32.")
    if input.device != output.device:
        raise RuntimeError("Input and output must reside on the same CUDA device.")
    if total > input.numel():
        raise RuntimeError("`total exceeds the number of elements in input.")
    if total > output.numel():
        raise RuntimeError("`total` exceeds the number of elements in output.")

    # Flatten to 1‑D contiguous buffers for Triton
    input_flat = input.reshape(-1).contiguous()
    output_flat = output.reshape(-1).contiguous()

    # Determine if we need to copy back to the original output tensor
    need_copy_back = output_flat.data_ptr() != output.data_ptr()

    # configuration: one program per BLOCK_SIZE elements
    grid = (triton.cdiv(total, BLOCK_SIZE),)

    # Launch kernel
    _triton_kernel_impl[grid](
        input_flat,
        output_flat,
        total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.cuda.synchronize()

    # Copy back if a temporary buffer was used
    if need_copy_back:
        output.copy_(output_flat.view(output.shape))