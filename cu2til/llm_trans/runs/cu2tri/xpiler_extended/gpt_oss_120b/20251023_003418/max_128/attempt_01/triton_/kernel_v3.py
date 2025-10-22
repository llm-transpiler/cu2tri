import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr,          # *float32
    output_ptr,         # *float32
    rows: tl.int32,
    inner: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute per-column maximum across `rows` rows.
    Each program processes a contiguous block of `BLOCK_SIZE` columns.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < inner

    # Initialize reduction value to -inf (equivalent to -FLT_MAX)
    max_val = tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32)

    # Loop over rows
    r = tl.zeros([], dtype=tl.int32)  # scalar row index
    while r < rows:
        # Linear index for the current row and each column offset
        idx = r * inner + offsets
        # Load with mask; out‑of‑range lanes get -inf
        val = tl.load(input_ptr + idx, mask=mask, other=-float("inf"))
        max_val = tl.maximum(max_val, val)
        r = r + 1

    # Write the result (masked store)
    tl.store(output_ptr + offsets, max_val, mask=mask)


def triton_kernel(input: torch.Tensor, output: torch.Tensor, rows: int, inner: int):
    """
    Triton implementation of the reduction kernel.
    Mirrors the CUDA signature:
        reduction_kernel(const float* input, float* output, int rows, int inner)

    Parameters
    ----------
    input : torch.Tensor
        Float32 tensor containing `rows * inner` elements.
        Can be either shape (rows, inner) or a flat 1‑D tensor of length rows*inner.
    output : torch.Tensor
        Tensor that receives the per‑column maxima.
        Accepts either a 1‑D tensor of shape (inner,) or a scalar (0‑D) tensor when inner == 1.
    rows : int
        Number of rows in the logical 2‑D view of `input`.
    inner : int
        Number of columns (inner dimension).
    """
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    assert input.is_cuda and output.is_cuda, "Both tensors must reside on a CUDA device"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 is supported"

    # Accept both 2‑D and flat 1‑D input layouts
    if input.dim() == 2:
        assert input.shape == (rows, inner), f"Expected input shape ({rows}, {inner}), got {input.shape}"
    elif input.dim() == 1:
        assert input.numel() == rows * inner, (
            f"Expected flat input of size {rows * inner}, got {input.numel()}"
        )
    else:
        raise AssertionError(f"Input must be 1‑D or 2‑D, got {input.dim()}‑D")

    # Output can be a scalar (0‑D) when inner == 1, or a 1‑D tensor of length `inner`
    if output.dim() == 0:
        assert output.numel() == inner, f"Expected scalar output with {inner} element(s), got {output.numel()}"
    elif output.dim() == 1:
        assert output.shape[0] == inner, f"Expected output shape ({inner},), got {output.shape}"
    else:
        raise AssertionError(f"Output must be 0‑D or 1‑D, got {output.dim()}‑D")

    # Ensure contiguous memory for pointer arithmetic
    input_contig = input if input.is_contiguous() else input.contiguous()
    output_contig = output if output.is_contiguous() else output.contiguous()

    # -------------------------------------------------------------------------
    # Kernel launch
    # -------------------------------------------------------------------------
    BLOCK_SIZE = 256
    grid = ((inner + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _triton_kernel_impl[grid](
        input_contig,
        output_contig,
        rows,
        inner,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )