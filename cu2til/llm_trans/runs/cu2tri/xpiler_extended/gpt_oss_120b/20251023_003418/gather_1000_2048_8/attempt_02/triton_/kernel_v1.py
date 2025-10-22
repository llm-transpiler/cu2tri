import torch
import triton
import triton.language as tl

# Block size matches the original CUDA configuration.
BLOCK_SIZE = 256

@triton.jit
def _triton_kernel_impl(
    params_ptr,      # float*   (dim0*dim1*dim2)
    indices_ptr,     # int64*   (num_indices)
    output_ptr,      # float*   (dim0*dim1*num_indices)
    dim0,            # int64
    dim1,            # int64
    dim2,            # int64
    num_indices,     # int64
    total,           # int64   (dim0*dim1*num_indices)
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel that reproduces the gather logic of the original CUDA kernel.
    """
    # -------------------------------------------------------------------------
    # Compute a 1‑D index for each thread.
    # -------------------------------------------------------------------------
    pid = tl.program_id(0).to(tl.int64)                     # block id
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE, dtype=tl.int64)
    mask = offsets < total                                    # guard against OOB

    # -------------------------------------------------------------------------
    # Decompose the linear index into (i0, i1, n) coordinates.
    # -------------------------------------------------------------------------
    stride = dim1 * num_indices                               # dim1 * num_indices
    i0 = offsets // stride
    rem = offsets % stride
    i1 = rem // num_indices
    n = rem % num_indices

    # -------------------------------------------------------------------------
    # Load the source index for this element.
    # -------------------------------------------------------------------------
    source = tl.load(indices_ptr + n, mask=mask, other=0)    # int64

    # -------------------------------------------------------------------------
    # Determine whether the source lies within the valid range [0, dim2).
    # -------------------------------------------------------------------------
    valid_source = (source >= 0) & (source < dim2)

    # -------------------------------------------------------------------------
    # Compute the flat offset into the params tensor:
    #   params[(i0 * dim1 + i1) * dim2 + source]
    # -------------------------------------------------------------------------
    param_offset = ((i0 * dim1 + i1) * dim2 + source)

    # -------------------------------------------------------------------------
    # Load the value from params when the source is valid; otherwise return 0.0.
    # -------------------------------------------------------------------------
    value = tl.load(
        params_ptr + param_offset,
        mask=valid_source & mask,
        other=0.0
    )

    # -------------------------------------------------------------------------
    # Write the result to the output tensor.
    # -------------------------------------------------------------------------
    tl.store(output_ptr + offsets, value, mask=mask)


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
    Entry‑point that launches the Triton gather kernel.

    Parameters
    ----------
    params : torch.Tensor
        Float32 tensor containing the source data. Expected shape is
        (dim0, dim1, dim2) or any contiguous view with the same element count.
    indices : torch.Tensor
        Int64 tensor of length ``num_indices`` holding the indices to gather.
    output : torch.Tensor
        Float32 tensor that will receive the gathered values. Expected shape is
        (dim0, dim1, num_indices) or any contiguous view with
        ``dim0*dim1*num_indices`` elements.
    dim0, dim1, dim2 : int
        Dimensions of the ``params`` tensor.
    num_indices : int
        Number of indices to gather (length of ``indices``).
    """
    # -------------------------------------------------------------------------
    # Basic sanity checks.
    # -------------------------------------------------------------------------
    assert params.is_cuda and indices.is_cuda and output.is_cuda, \
        "All tensors must reside on the same CUDA device."
    assert params.dtype == torch.float32, "params must be a float32 tensor."
    assert output.dtype == torch.float32, "output must be a float32 tensor."
    assert indices.dtype == torch.int64, "indices must be an int64 tensor."
    assert params.is_contiguous(), "params must be contiguous."
    assert indices.is_contiguous(), "indices must be contiguous."
    assert output.is_contiguous(), "output must be contiguous."

    # -------------------------------------------------------------------------
    # Compute total number of output elements and launch configuration.
    # -------------------------------------------------------------------------
    total = dim0 * dim1 * num_indices
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # -------------------------------------------------------------------------
    # Launch the Triton kernel.
    # -------------------------------------------------------------------------
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
    # Optional synchronization for debugging; can be removed for async execution.
    # torch.cuda.synchronize()