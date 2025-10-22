import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing the CUDA kernel logic
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    # Pointers
    attention_weights_ptr,  # float *
    output_ptr,            # float *
    sampling_locations_ptr,  # float *
    value_ptr,               # float *
    value_level_start_index_ptr,  # int *
    value_spatial_shapes_ptr,     # int *
    # Compile‑time constants
    BLOCK_SIZE: tl.constexpr,   # number of threads per block (64)
):
    # ------------------------------------------------------------------
    # Program IDs (block indices)
    # ------------------------------------------------------------------
    pid_x = tl.program_id(0)  # blockIdx.x  (query index)
    pid_y = tl.program_id(1)  # blockIdx.y  (unused, always 0)
    pid_z = tl.program_id(2)  # blockIdx.z  (head index, 0..7)

    # ------------------------------------------------------------------
    # Thread indices inside the block
    # ------------------------------------------------------------------
    # Each thread processes 4 consecutive channels
    thread_idx = tl.arange(0, BLOCK_SIZE, dtype=tl.int64)          # (BLOCK_SIZE,)
    channel_range = tl.arange(0, 4, dtype=tl.int64)                # (4,)
    channel_offset = thread_idx * 4                                 # (BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Accumulator for the output (4 channels per thread)
    # ------------------------------------------------------------------
    attention_sum = tl.zeros((BLOCK_SIZE, 4), dtype=tl.float32)

    # ------------------------------------------------------------------
    # Constants used in the indexing arithmetic
    # ------------------------------------------------------------------
    LEVEL_STRIDE = 2048   # 8 heads * 256 channels
    HEAD_STRIDE = 256     # 256 channels per head
    CHANNEL_BLOCK = 4

    # ------------------------------------------------------------------
    # Main loops: 4 levels (i) × 4 sampling points per level (k)
    # ------------------------------------------------------------------
    for i in range(4):
        # Load spatial shape for level i: height, width (int32)
        height = tl.load(value_spatial_shapes_ptr + 2 * i, dtype=tl.int32)
        width  = tl.load(value_spatial_shapes_ptr + 2 * i + 1, dtype=tl.int32)

        # Load level start offset (int32)
        level_start = tl.load(value_level_start_index_ptr + i, dtype=tl.int32)

        # Pre‑compute the constant part of the value offset for this level & head
        base_value_offset = tl.cast(level_start, tl.int64) * LEVEL_STRIDE \
                            + tl.cast(pid_z, tl.int64) * HEAD_STRIDE

        for k in range(4):
            # ------------------------------------------------------------------
            # Load the (y, x) sampling location pair (float)
            #   sampling_locations layout:
            #   [query, head, level, point, 2]  flattened as:
            #   idx = pid_x * 256 + pid_z * 32 + i * 8 + k * 2
            # ------------------------------------------------------------------
            base_loc_idx = pid_x * 256 + pid_z * 32 + i * 8 + k * 2
            xy_y = tl.load(sampling_locations_ptr + base_loc_idx, dtype=tl.float32)      # y
            xy_x = tl.load(sampling_locations_ptr + base_loc_idx + 1, dtype=tl.float32)  # x

            # ------------------------------------------------------------------
            # Convert normalized coordinates to absolute grid coordinates
            #   xy_grid = xy * size - 0.5
            # ------------------------------------------------------------------
            xy_grid_x = xy_x * tl.cast(height, tl.float32) - 0.5
            xy_grid_y = xy_y * tl.cast(width,  tl.float32) - 0.5

            # ------------------------------------------------------------------
            # Integer corner indices (floor and ceil)
            # ------------------------------------------------------------------
            x0 = tl.cast(tl.floor(xy_grid_x), tl.int32)
            x1 = x0 + 1
            y0 = tl.cast(tl.floor(xy_grid_y), tl.int32)
            y1 = y0 + 1

            # ------------------------------------------------------------------
            # Masks for the four bilinear corners (validity check)
            # ------------------------------------------------------------------
            mask00 = (x0 >= 0) & (x0 < height) & (y0 >= 0) & (y0 < width)
            mask01 = (x0 >= 0) & (x0 < height) & (y1 >= 0) & (y1 < width)
            mask10 = (x1 >= 0) & (x1 < height) & (y0 >= 0) & (y0 < width)
            mask11 = (x1 >= 0) & (x1 < height) & (y1 >= 0) & (y1 < width)

            # ------------------------------------------------------------------
            # Helper to compute the value offset for a corner
            # ------------------------------------------------------------------
            def corner_offset(x_idx, y_idx):
                # linear index inside the (H, W) plane
                xy_index = tl.cast(x_idx, tl.int64) * tl.cast(width, tl.int64) \
                           + tl.cast(y_idx, tl.int64)          # (BLOCK_SIZE,)
                return base_value_offset + xy_index * LEVEL_STRIDE \
                       + channel_offset[:, None] + channel_range[None, :]

            # ------------------------------------------------------------------
            # Load corner values (4 channels per thread), zero‑filled where out‑of‑bounds
            # ------------------------------------------------------------------
            corner00 = tl.load(value_ptr + corner_offset(x0, y0),
                               mask=mask00[:, None], other=0.0)
            corner01 = tl.load(value_ptr + corner_offset(x0, y1),
                               mask=mask01[:, None], other=0.0)
            corner10 = tl.load(value_ptr + corner_offset(x1, y0),
                               mask=mask10[:, None], other=0.0)
            corner11 = tl.load(value_ptr + corner_offset(x1, y1),
                               mask=mask11[:, None], other=0.0)

            # ------------------------------------------------------------------
            # Bilinear interpolation weights (scalar per thread, broadcasted)
            # ------------------------------------------------------------------
            w00 = (tl.cast(x1, tl.float32) - xy_grid_x) * (tl.cast(y1, tl.float32) - xy_grid_y)
            w01 = (tl.cast(x1, tl.float32) - xy_grid_x) * (xy_grid_y - tl.cast(y0, tl.float32))
            w10 = (xy_grid_x - tl.cast(x0, tl.float32)) * (tl.cast(y1, tl.float32) - xy_grid_y)
            w11 = (xy_grid_x - tl.cast(x0, tl.float32)) * (xy_grid_y - tl.cast(y0, tl.float32))

            # ------------------------------------------------------------------
            # Weighted sum of the four corners
            # ------------------------------------------------------------------
            weighted = (corner00 * w00[:, None] +
                        corner01 * w01[:, None] +
                        corner10 * w10[:, None] +
                        corner11 * w11[:, None])

            # ------------------------------------------------------------------
            # Load the scalar attention weight for this (i, k) pair
            #   layout: [query, head, level, point] flattened as:
            #   idx = pid_x * 128 + pid_z * 16 + i * 4 + k
            # ------------------------------------------------------------------
            att_idx = pid_x * 128 + pid_z * 16 + i * 4 + k
            att_weight = tl.load(attention_weights_ptr + att_idx, dtype=tl.float32)

            # ------------------------------------------------------------------
            # Accumulate into the per‑thread output accumulator
            # ------------------------------------------------------------------
            attention_sum += weighted * att_weight

    # ------------------------------------------------------------------
    # Write the accumulated result to the output tensor
    #   layout: [query, head, channel] flattened as:
    #   idx = pid_x * 2048 + pid_z * 256 + thread_idx * 4 + channel_range
    # ------------------------------------------------------------------
    out_offset = pid_x * 2048 + pid_z * 256 \
                 + channel_offset[:, None] + channel_range[None, :]
    tl.store(output_ptr + out_offset, attention_sum)


# ----------------------------------------------------------------------
# Python wrapper that mimics the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(
    value: torch.Tensor,                     # float32, shape = (num_levels, ?, ?, 8, 256) flattened
    value_spatial_shapes: torch.Tensor,      # int32, shape = (num_levels * 2,)
    level_start_index: torch.Tensor,         # int32, shape = (num_levels,)
    sampling_locations: torch.Tensor,        # float32, shape = (lq * 256,)
    attention_weights: torch.Tensor,         # float32, shape = (lq * 128,)
    output: torch.Tensor                     # float32, shape = (lq * 2048,)
):
    """
    Triton implementation of the CUDA kernel `_cuda_kernel_impl`.
    The argument order matches the original host function `cuda_kernel`.
    All tensors must reside on the same CUDA device.
    """
    # ------------------------------------------------------------------
    # Basic sanity checks
    # ------------------------------------------------------------------
    assert value.is_cuda and sampling_locations.is_cuda and attention_weights.is_cuda \
           and output.is_cuda and value_spatial_shapes.is_cuda and level_start_index.is_cuda, \
           "All tensors must be CUDA tensors."
    assert value.dtype == torch.float32 and sampling_locations.dtype == torch.float32 \
           and attention_weights.dtype == torch.float32 and output.dtype == torch.float32, \
           "Float tensors must be torch.float32."
    assert value_spatial_shapes.dtype == torch.int32 and level_start_index.dtype == torch.int32, \
           "Index tensors must be torch.int32."

    # Ensure contiguous layout for pointer arithmetic
    value = value.contiguous()
    value_spatial_shapes = value_spatial_shapes.contiguous()
    level_start_index = level_start_index.contiguous()
    sampling_locations = sampling_locations.contiguous()
    attention_weights = attention_weights.contiguous()
    output = output.contiguous()

    # ------------------------------------------------------------------
    # Derive grid dimensions
    #   - Number of heads is fixed to 8 (as in the original kernel)
    #   - Number of queries (lq) can be inferred from the output size:
    #       output size = lq * 2048  (2048 = 8 heads * 256 channels)
    # ------------------------------------------------------------------
    HEADS = 8
    CHANNELS_PER_HEAD = 256
    lq = output.numel() // (HEADS * CHANNELS_PER_HEAD)
    grid = (lq, 1, HEADS)          # (blockIdx.x, blockIdx.y, blockIdx.z)
    block = (64,)                   # BLOCK_SIZE = 64 (matches __launch_bounds__(64))

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
        BLOCK_SIZE=64,
        num_warps=4,                # reasonable default for 64‑thread blocks
    )
    # Synchronize to make sure the kernel has finished (optional)
    torch.cuda.synchronize()
    return output