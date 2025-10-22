import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr,          # *float32
    output_ptr,         # *float32
    rows: tl.constexpr, # compile‑time constant (unrolled loop)
    inner,              # runtime scalar (int)
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute per‑column minimum across `rows` rows.
    Each program instance processes a block of `BLOCK_SIZE` columns.
    """
    pid = tl.program_id(0)  # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # column indices for this block
    mask = offsets < inner   # guard for out‑of‑range columns

    # Large initial value (equivalent to FLT_MAX)
    init_val = 3.402823466e+38
    min_val = tl.full([BLOCK_SIZE], init_val, dtype=tl.float32)

    # Unrolled loop over rows (rows is a compile‑time constant)
    for r in range(rows):
        idx = r * inner + offsets               # linear index of element (r, col)
        cur = tl.load(input_ptr + idx, mask=mask, other=init_val)
        min_val = tl.minimum(min_val, cur)

    # Write the per‑column minima to output
    tl.store(output_ptr + offsets, min_val, mask=mask)


def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    inner: int
):
    """
    Wrapper that launches the Triton kernel.
    Mirrors the signature of the original CUDA kernel:
        reduction_kernel(const float* input, float* output, int rows, int inner)
    """
    # Basic sanity checks
    assert input.is_cuda and output.is_cuda, "Both tensors must reside on CUDA"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 supported"
    assert input.shape == (rows, inner), f"input shape mismatch: expected ({rows}, {inner})"
    assert output.shape == (inner,), f"output shape mismatch: expected ({inner},)"

    # Ensure contiguous layout for pointer arithmetic
    input = input.contiguous()
    output = output.contiguous()

    BLOCK_SIZE = 256  # matches the original CUDA thread block size
    grid = ((inner + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the kernel; `rows` is a compile‑time constant (constexpr)
    _triton_kernel_impl[grid](
        input,
        output,
        rows,
        inner,
        BLOCK_SIZE=BLOCK_SIZE
    )