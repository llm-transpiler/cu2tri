import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
BLOCK_SIZE = 256  # Matches the original CUDA block size

# -------------------------------------------------------------------------
# Triton kernel implementation
# -------------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    params_ptr,      # float*   (dim0 * dim1 * dim2)
    indices_ptr,     # int64*   (num_indices)
    output_ptr,      # float*   (dim0 * dim1 * num_indices)
    dim0,            # int64
    dim1,            # int64
    dim2,            # int64
    num_indices,     # int64
    total,           # int64   (dim0 * dim1 * num_indices)
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel that reproduces the gather logic of the original CUDA kernel.
    """
    # -----------------------------------------------------------------
    # 1‑D thread index
    # -----------------------------------------------------------------
    pid = tl.program_id(0).to(tl.int64)                     # block id
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < total                                    # out‑of‑bounds guard

    # -----------------------------------------------------------------
    # Decompose linear index into (i0, i1, n)
    # -----------------------------------------------------------------
    stride = dim1 * num_indices
    i0 = offsets // stride
    rem = offsets % stride
    i1 = rem // num_indices
    n = rem % num_indices

    # -----------------------------------------------------------------
    # Load the source index for this element
    # -----------------------------------------------------------------
    source = tl.load(indices_ptr + n, mask=mask, other=0)    # int64

    # -----------------------------------------------------------------
    # Check source validity
    # -----------------------------------------------------------------
    valid_source = (source >= 0) & (source < dim2)

    # -----------------------------------------------------------------
    # Compute flat offset into params: params[(i0 * dim1 + i1) * dim2 + source]
    # -----------------------------------------------------------------
    param_offset = ((i0 * dim1 + i1) * dim2 + source)

    # -----------------------------------------------------------------
    # Load value from params when source is valid; otherwise 0.0
    # -----------------------------------------------------------------
    value = tl.load(
        params_ptr + param_offset,
        mask=valid_source & mask,
        other=0.0
    )

    # -----------------------------------------------------------------
    # Write result to output
    # -----------------------------------------------------------------
    tl.store(output_ptr + offsets, value, mask=mask)


# -------------------------------------------------------------------------
# Wrapper entry point (mirrors the original CUDA kernel signature)
# -------------------------------------------------------------------------
def triton_kernel(
    params: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    dim0: int,
    dim1: int,
    dim2: int,
    num_indices: int
):
    """
    Launches the Triton gather kernel.

    Parameters
    ----------
    params : torch.Tensor
        Float32 tensor of shape (dim0, dim1, dim2) (contiguous, CUDA).
    indices : torch.Tensor
        Int64 tensor of length ``num_indices`` (contiguous, CUDA).
    output : torch.Tensor
        Float32 tensor of shape (dim0, dim1, num_indices) (contiguous, CUDA).
    dim0, dim1, dim2, num_indices : int
        Dimensions matching the CUDA implementation.
    """
    # -----------------------------------------------------------------
    # Basic sanity checks
    # -----------------------------------------------------------------
    assert params.is_cuda and indices.is_cuda and output.is_cuda, \
        "All tensors must reside on the same CUDA device."
    assert params.dtype == torch.float32, "params must be a float32 tensor."
    assert indices.dtype == torch.int64, "indices must be an int64 tensor."
    assert output.dtype == torch.float32, "output must be a float32 tensor."
    assert params.is_contiguous(), "params must be contiguous."
    assert indices.is_contiguous(), "indices must be contiguous."
    assert output.is_contiguous(), "output must be contiguous."

    # -----------------------------------------------------------------
    # Compute launch configuration
    # -----------------------------------------------------------------
    total = dim0 * dim1 * num_indices
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # -----------------------------------------------------------------
    # Kernel launch
    # -----------------------------------------------------------------
    _triton_kernel_impl[grid](
        params,
        indices,
        output,
        dim0,
        dim1,
        dim2,
        num_indices,
        total,
        BLOCK_SIZE=BLOCK_SIZE
    )
    # Uncomment for debugging / precise timing
    # torch.cuda.synchronize()