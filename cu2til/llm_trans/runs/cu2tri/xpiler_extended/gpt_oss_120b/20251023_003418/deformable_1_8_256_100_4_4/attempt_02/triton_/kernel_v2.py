import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing the original CUDA kernel logic
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    attention_weights_ptr,          # float *
    output_ptr,                    # float *
    sampling_locations_ptr,        # float *
    value_ptr,                     # float *
    value_level_start_index_ptr,   # int *
    value_spatial_shapes_ptr,      # int *
    BLOCK_SIZE: tl.constexpr,      # compile‑time constant (64)
):
    # --------------------------------------------------------------
    # Program IDs (block indices)
    # --------------------------------------------------------------
    pid_x = tl.program_id(0)  # blockIdx.x  (query index)
    pid_y = tl.program_id(1)  # blockIdx.y  (unused)
    pid_z = tl.program_id(2)  # blockIdx.z  (head index)

    # --------------------------------------------------------------
    # Thread indices inside the block
    # --------------------------------------------------------------
    thread_idx = tl.arange(0, BLOCK_SIZE, dtype=tl.int64)   # (BLOCK_SIZE,)
    channel_range = tl.arange(0, 4, dtype=tl.int64)         # (4,)
    channel_offset = thread_idx * 4                         # (BLOCK_SIZE,)

    # --------------------------------------------------------------
    # Accumulator for the per‑thread output (4 channels)
    # --------------------------------------------------------------
    attention_sum = tl.zeros((BLOCK_SIZE, 4), dtype=tl.float32)

    # --------------------------------------------------------------
    # Constants used in the indexing arithmetic
    # --------------------------------------------------------------
    LEVEL_STRIDE = 2048   # heads * channels_per_head
    HEAD_STRIDE = 256     # channels per head

    # --------------------------------------------------------------
    # Main loops: 4 levels (i) × 4 sampling points per level (k)
    # --------------------------------------------------------------
    for i in range(4):
        # Load spatial shape for level i: height, width (int32)
        height = tl.load(value_spatial_shapes_ptr + 2 * i, dtype=tl.int32)
        width  = tl.load(value_spatial_shapes_ptr + 2 * i + 1, dtype=tl.int32)

        # Load start offset for level i (int32)
        level_start = tl.load(value_level_start_index_ptr + i, dtype=tl.int32)

        # Base offset for this level and head
        base_value_offset = tl.cast(level_start, tl.int64) * LEVEL_STRIDE \
                            + tl.cast(pid_z, tl.int64) * HEAD_STRIDE

        for k in range(4):
            # ----------------------------------------------------------
            # Load the (y, x) sampling location pair (float)
            # ----------------------------------------------------------
            base_loc_idx = pid_x * 256 + pid_z * 32 + i * 8 + k * 2
            xy_y = tl.load(sampling_locations_ptr + base_loc_idx, dtype=tl.float32)      # y
            xy_x = tl.load(sampling_locations_ptr + base_loc_idx + 1, dtype=tl.float32)  # x

            # ----------------------------------------------------------
            # Convert normalized coordinates to absolute grid coordinates
            # ----------------------------------------------------------
            xy_grid_x = xy_x * tl.cast(height, tl.float32) - 0.5
            xy_grid_y = xy_y * tl.cast(width,  tl.float32) - 0.5

            # ----------------------------------------------------------
            # Integer corner indices (floor and ceil)
            # ----------------------------------------------------------
            x0 = tl.cast(tl.floor(xy_grid_x), tl.int32)
            x1 = x0 + 1
            y0 = tl.cast(tl.floor(xy_grid_y), tl.int32)
            y1 = y0 + 1

            # ----------------------------------------------------------
            # Masks for the four bilinear corners (validity check)
            # ----------------------------------------------------------
            mask00 = (x0 >= 0) & (x0 < height) & (y0 >= 0) & (y0 < width)
            mask01 = (x0 >= 0) & (x0 < height) & (y1 >= 0) & (y1 < width)
            mask10 = (x1 >= 0) & (x1 < height) & (y0 >= 0) & (y0 < width)
            mask11 = (x1 >= 0) & (x1 < height) & (y1 >= 0) & (y1 < width)

            # ----------------------------------------------------------
            # Helper to compute a safe offset for a corner
            # ----------------------------------------------------------
            def corner_offset(x_idx, y_idx):
                # Clamp indices to the valid range to avoid OOB address computation
                x_clamped = tl.where(x_idx < 0, 0,
                               tl.where(x_idx >= height, height - 1, x_idx))
                y_clamped = tl.where(y_idx < 0, 0,
                               tl.where(y_idx >= width, width - 1, y_idx))
                xy_index = tl.cast(x_clamped, tl.int64) * tl.cast(width, tl.int64) \
                           + tl.cast(y_clamped, tl.int64)          # (BLOCK_SIZE,)
                return base_value_offset + xy_index * LEVEL_STRIDE \
                       + channel_offset[:, None] + channel_range[None, :]

            # ----------------------------------------------------------
            # Load corner values (4 channels per thread), zero‑filled where out‑of‑bounds
            # ----------------------------------------------------------
            corner00 = tl.load(value_ptr + corner_offset(x0, y0),
                               mask=mask00[:, None], other=0.0)
            corner01 = tl.load(value_ptr + corner_offset(x0, y1),
                               mask=mask01[:, None], other=0.0)
            corner10 = tl.load(value_ptr + corner_offset(x1, y0),
                               mask=mask10[:, None], other=0.0)
            corner11 = tl.load(value_ptr + corner_offset(x1, y1),
                               mask=mask11[:, None], other=0.0)

            # ----------------------------------------------------------
            # Bilinear interpolation weights (scalar per thread, broadcasted)
            # ----------------------------------------------------------
            w00 = (tl.cast(x1, tl.float32) - xy_grid_x) * (tl.cast(y1, tl.float32) - xy_grid_y)
            w01 = (tl.cast(x1, tl.float32) - xy_grid_x) * (xy_grid_y - tl.cast(y0, tl.float32))
            w10 = (xy_grid_x - tl.cast(x0, tl.float32)) * (tl.cast(y1, tl.float32) - xy_grid_y)
            w11 = (xy_grid_x - tl.cast(x0, tl.float32)) * (xy_grid_y - tl.cast(y0, tl.float32))

            # ----------------------------------------------------------
            # Weighted sum of the four corners
            # ----------------------------------------------------------
            weighted = (corner00 * w00[:, None] +
                        corner01 * w01[:, None] +
                        corner10 * w10[:, None] +
                        corner11 * w11[:, None])

            # ----------------------------------------------------------
            # Load the scalar attention weight for this (i, k) pair
            # ----------------------------------------------------------
            att_idx = pid_x * 128 + pid_z * 16 + i * 4 + k
            att_weight = tl.load(attention_weights_ptr + att_idx, dtype=tl.float32)

            # ----------------------------------------------------------
            # Accumulate into the per‑thread output accumulator
            # ----------------------------------------------------------
            attention_sum += weighted * att_weight

    # --------------------------------------------------------------
    # Write the accumulated result to the output tensor
    # --------------------------------------------------------------
    out_offset = pid_x * 2048 + pid_z * 256 \
                 + channel_offset[:, None] +_range[None, :]
    tl.store(output_ptr + out_offset, attention_sum)


# ----------------------------------------------------------------------
# Python wrapper that mimics the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    output: torch.Tensor,
):
    """
    Triton implementation of the original CUDA kernel.
    Argument order matches the original host function:
        value, value_spatial_shapes, level_start_index,
        sampling_locations, attention_weights, output
    All tensors must be CUDA tensors with the correct dtypes.
    """
    # ------------------------------------------------------------------
    # Basic sanity checks
    # ------------------------------------------------------------------
    for name, tensor, expected_dtype in [
        ("value", value, torch.float32),
        ("sampling_locations", sampling_locations, torch.float32),
        ("attention_weights", attention_weights, torch.float32),
        ("output", output, torch.float32),
        ("value_spatial_shapes", value_spatial_shapes, torch.int32),
        ("level_start_index", level_start_index, torch.int32),
    ]:
        assert tensor.is_cuda, f"{name} must be a CUDA tensor."
 tensor.dtype == expected_dtype, f"{name} must have dtype {expected_dtype}."
        assert tensor.is_contiguous(), f"{name} must be contiguous."

    # ------------------------------------------------------------------
    # Derive dimensions (the original kernel assumes 8 heads, 256 channels per head)
    # ------------------------------------------------------------------
    CHANNELS_PER_HEAD = 256
    HEADS = 8
    total_output = output.numel()
    assert total_output % (HEADS * CHANNELS_PER_HEAD) == 0, (
        "Output size is not compatible with 8 heads and 256 channels per head."
    )
    lq = total_output // (HEADS * CHANNELS_PER_HEAD)

    # ------------------------------------------------------------------
    # Grid configuration: (blockIdx.x, blockIdx.y, blockIdx.z)
    # ------------------------------------------------------------------
    grid = (lq, 1, HEADS)   # (queries, 1, heads)

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        attention_weights,
        output,
        sampling_locations,
        value,
        level_start_index,
        value_spatial_shapes,
        BLOCK_SIZE=64,          # matches __launch_bounds__(64)
        num_warps=2,           # 64 threads = 2 warps
    )
    torch.cuda.synchronize()
    return output


# Alias for compatibility with the original CUDA host function name
cuda_kernel = triton_kernel

__all__ = ["triton_kernel", "cuda_kernel"]