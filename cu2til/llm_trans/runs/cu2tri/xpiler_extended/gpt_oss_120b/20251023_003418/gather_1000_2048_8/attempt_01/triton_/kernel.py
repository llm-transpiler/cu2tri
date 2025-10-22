import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementation (mirrors the original CUDA gather kernel)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    params_ptr,      # float*   (dim0 * dim1 * dim2)
    indices_ptr,     # int64*   (num_indices)
    output_ptr,      # float*   (dim0 * dim1 * num_indices)
    dim0,            # int32
    dim1,            # int32
    dim2,            # int32
    num_indices,     # int32
    total,           # int64   (dim0 * dim1 * num_indices)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton implementation of the gather kernel.
    """
    # Program (block) identifier as int64
    pid = tl.int64(tl.program_id(0))

    # Flat offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE, dtype=tl.int64)
    mask = offsets < total  # valid output elements

    # Cast dimensions to int64 for arithmetic
    dim1_i64 = tl.int64(dim1)
    dim2_i64 = tl.int64(dim2)
    num_idx_i64 = tl.int64(num_indices)

    # Decompose flat index into (i0, i1, n)
    stride = dim1_i64 * num_idx_i64
    i0 = offsets // stride
    rem = offsets % stride
    i1 = rem // num_idx_i64
    n = rem % num_idx_i64

    # Load source index from the indices vector
    src = tl.load(indices_ptr + n, mask=mask, other=0)  # int64

    # Validity check for the source index
    src_valid = (src >= 0) & (src < dim2_i64)

    # Safe source index for address calculation (masked later)
    src_safe = tl.where(src_valid, src, tl.zeros_like(src))

    # Compute flat offset into the params tensor
    param_offset = ((i0 * dim1_i64 + i1) * dim2_i64) + src_safe

    # Load the value from params only when the source index is valid
    val = tl.load(
        params_ptr + param_offset,
        mask=mask & src_valid,
        other=0.0,
    )  # float32

    # Write the result
    tl.store(output_ptr + offsets, val, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(
    params: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    dim0: int,
    dim1: int,
    dim2: int,
    num_indices: int,
):
    """
    Entry point that mirrors the original CUDA kernel signature.
    Parameters
    ----------
    params   – float32 tensor of shape (dim0, dim1, dim2)
    indices  – int64   tensor of length >= num_indices
    output   – float32 tensor of length dim0 * dim1 * num_indices
    dim0, dim1, dim2, num_indices – integer dimensions
    """
    # ------------------------------------------------------------------
    # Validation (mirrors expectations of the original CUDA implementation)
    # ------------------------------------------------------------------
    if not (params.is_cuda and indices.is_cuda and output.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if not (params.dtype == torch.float32 and indices.dtype == torch.int64 and output.dtype == torch.float32):
        raise RuntimeError("Expected params/output as float32 and indices as int64")
    if not (params.is_contiguous() and indices.is_contiguous() and output.is_contiguous()):
        raise RuntimeError("All tensors must be contiguous")
    if params.shape != (dim0, dim1, dim2):
        raise RuntimeError(
            f"params shape {params.shape} does not match (dim0,dim1,dim2)=({dim0},{dim1},{dim2})"
        )
    if indices.shape[0] < num_indices:
        raise RuntimeError("indices tensor is shorter than num_indices")
    total = dim0 * dim1 * num_indices
    if output.numel() < total:
        raise RuntimeError("output tensor is too small for the requested total size")
    if total == 0:
        return  # nothing to do

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    BLOCK_SIZE = 256
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _triton_kernel_impl[grid](
        params,
        indices,
        output,
        dim0,
        dim1,
        dim2,
        num_indices,
        total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Optional synchronization for debugging; uncomment if needed
    # torch.cuda.synchronize()