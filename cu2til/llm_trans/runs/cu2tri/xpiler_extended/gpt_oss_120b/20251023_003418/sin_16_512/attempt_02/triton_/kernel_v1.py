import torch
import triton
import triton.language as tl

# Compile‑time block size (matches the CUDA version's 256 threads per block)
BLOCK_SIZE = 256

@triton.jit
def _triton_kernel_impl(input_ptr, output_ptr, total, BLOCK_SIZE: tl.constexpr):
    """Compute sin of each element in `input_ptr` and write to `output_ptr`."""
    pid = tl.program_id(0)                     # program (block) index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # linear indices
    mask = offsets < total                      # guard against out‑of‑bounds

    # Load input values (masked load returns 0.0 for OOB elements)
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Compute sine
    y = tl.sin(x)
    # Store results (masked store)
    tl.store(output_ptr + offsets, y, mask=mask)

def triton_kernel(input: torch.Tensor, output: torch.Tensor, total: int):
    """
    Triton entry point mirroring the original CUDA kernel signature.

    Parameters
    ----------
    input : torch.Tensor
        1‑D float32 CUDA tensor containing the input values.
    output : torch.Tensor
        1‑D float32 CUDA tensor where the sine results will be written.
    total : int
        Number of elements to process.
    """
    # Basic validation (mirrors the expectations of the CUDA kernel)
    if not input.is_cuda or not output.is_cuda:
        raise RuntimeError("Both input and output must be CUDA tensors.")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("Both input and output must be of type torch.float32.")
    if input.dim() != 1 or output.dim() != 1:
        raise RuntimeError("Both input and output must be 1‑dimensional.")
    if total > input.numel() or total > output.numel():
        raise RuntimeError("`total` exceeds the number of elements in input or output.")

    # Ensure contiguous layout for optimal memory access
    input = input.contiguous()
    output = output.contiguous()

    # Compute grid size: one program per BLOCK_SIZE elements
    grid = (triton.cdiv(total, BLOCK_SIZE),)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        output,
        total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Optional synchronization to guarantee completion before returning to Python
    torch.cuda.synchronize()