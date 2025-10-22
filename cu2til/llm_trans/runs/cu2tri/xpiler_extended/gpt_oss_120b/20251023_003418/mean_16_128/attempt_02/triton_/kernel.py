import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: performs column-wise reduction (average) over `rows`
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    input_ptr,          # tl.pointer to float32 input tensor (rows x inner)
    output_ptr,         # tl.pointer to float32 output tensor (inner)
    rows: tl.int32,     # number of rows to reduce over
    inner: tl.int32,    # inner dimension size (output length)
    BLOCK_SIZE: tl.constexpr  # compile‑time block size (threads per program)
):
    # Program ID identifies which slice of the output we are responsible for
    pid = tl.program_id(0)
    # Offsets for the elements this program will write (one per thread)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask out-of-range indices (when inner is not a multiple of BLOCK_SIZE)
    mask = offsets < inner

    # Accumulator for the sum of each column
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Dynamic loop over the rows dimension
    r = 0
    while r < rows:
        # Compute pointer to the start of row `r` for all columns in this block
        ptr = input_ptr + r * inner + offsets
        # Load values (masked loads return 0.0 for out‑of‑range threads)
        vals = tl.load(ptr, mask=mask, other=0.0)
        acc += vals
        r += 1

    # Compute the average
    avg = acc / rows
    # Write the result back to the output tensor
    tl.store(output_ptr + offsets, avg, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper: mirrors the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    inner: int
):
    """
    Triton implementation of the original CUDA reduction kernel.

    For each column index `idx` in [0, inner):
        output[idx] = (1/rows) * sum_{r=0}^{rows-1} input[r, idx]

    Arguments:
        input  : torch.Tensor of shape (rows, inner), dtype torch.float32, CUDA device.
        output : torch.Tensor of shape (inner,), dtype torch.float32, CUDA device.
        rows   : int, number of rows (must match input.shape[0]).
        inner  : int, inner dimension size (must match input.shape[1] and output.shape[0]).
    """
    # ------------------------------------------------------------------
    # Sanity checks (mirroring the expectations of the original kernel)
    # ------------------------------------------------------------------
    assert input.is_cuda and output.is_cuda, "Both tensors must reside on a CUDA device"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 tensors are supported"
    assert input.is_contiguous() and output.is_contiguous(), "Tensors must be contiguous"
    assert input.dim() == 2, "Input tensor must be 2‑dimensional (rows, inner)"
    assert output.dim() == 1, "Output tensor must be 1‑dimensional (inner)"
    assert input.shape[0] == rows, f"Input rows {input.shape[0]} do not match the provided rows argument {rows}"
    assert input.shape[1] == inner, f"Input inner dimension {input.shape[1]} does not match the provided inner argument {inner}"
    assert output.shape[0] == inner, f"Output size {output.shape[0]} does not match the provided inner argument {inner}"
    assert rows > 0 and inner > 0, "rows and inner must be positive integers"

    # ------------------------------------------------------------------
    # Kernel launch configuration
    # ------------------------------------------------------------------
    BLOCK_SIZE = 128  # Tunable: balances occupancy and register pressure
    grid = (triton.cdiv(inner, BLOCK_SIZE),)  # One program per BLOCK_SIZE columns

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        input,          # Triton automatically treats torch tensors as pointers
        output,
        rows,
        inner,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Optional synchronization for debugging (comment out in production)
    # torch.cuda.synchronize()