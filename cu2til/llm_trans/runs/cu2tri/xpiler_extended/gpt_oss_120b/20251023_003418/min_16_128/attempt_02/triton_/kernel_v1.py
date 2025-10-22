import torch
import triton
import triton.language as tl

# Maximum float32 value (same as FLT_MAX)
FLT_MAX = 3.402823466e+38

@triton.jit
def _triton_kernel_impl(
    input_ptr,          # *float32
    output_ptr,         # *float32
    rows: tl.constexpr, # compile‑time constant for loop unrolling
    inner,              # runtime scalar (int)
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute the per‑column minimum across `rows` rows.
    Each thread processes one column index `idx` in [0, inner).
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread indices
    mask = offsets < inner                      # guard for out‑of‑range columns

    # Initialise the reduction with a large value (FLT_MAX)
    min_val = tl.full([BLOCK_SIZE], FLT_MAX, dtype=tl.float32)

    # Unrolled loop over rows (rows is a compile‑time constant)
    for r in range(rows):
        # Linear index of element (r, idx) in a row‑major layout
        idx = r * inner + offsets
        cur = tl.load(input_ptr + idx, mask=mask, other=FLT_MAX)
        min_val = tl.minimum(min_val, cur)

    # Write the result back to output
    tl.store(output_ptr + offsets, min_val, mask=mask)


def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    inner: int
):
    """
    Wrapper that launches the Triton kernel.
    Parameters
    ----------
    input : torch.Tensor
        Tensor of shape (rows, inner), dtype torch.float32, on CUDA.
    output : torch.Tensor
        Tensor of shape (inner,), dtype torch.float32, on CUDA.
    rows : int
        Number of rows (must match input.shape[0]).
    inner : int
        Size of the inner dimension (must match input.shape[1]).
    """
    # Basic sanity checks
    assert input.is_cuda and output.is_cuda, "Both tensors must reside on CUDA"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 supported"
    assert input.shape == (rows, inner), f"input shape mismatch: expected ({rows}, {inner})"
    assert output.shape == (inner,), f"output shape mismatch: expected ({inner},)"

    # Ensure contiguous memory layout
    input = input.contiguous()
    output = output.contiguous()

    BLOCK_SIZE = 
    # Compute grid dimensions (one‑dimensional grid)
    grid = ((inner + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the kernel; rows is a compile‑time constant (constexpr)
    _triton_kernel_impl[grid](
        input,
        output,
        rows,
        inner,
        BLOCK_SIZE=BLOCK_SIZE
    )