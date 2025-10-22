import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel (mirrors the original CUDA implementation)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    attention_weights_ptr,
    output_ptr,
    sampling_locations_ptr,
    value_ptr,
    level_start_index_ptr,
    value_spatial_shapes_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = tl.program_id(0)  # blockIdx.x
    pid_z = tl.program_id(2)  # blockIdx.z

    # thread index within the block (0 .. BLOCK_SIZE-1)
    tid = tl.arange(0, BLOCK_SIZE)  # (BLOCK_SIZE,)

    # accumulator for the 4 output channels per thread
    att_sum = tl.zeros((BLOCK_SIZE, 4), dtype=tl.float32)

    # offsets for the 4 contiguous values each thread writes/reads
    col_offsets = tl.arange(0, 4)[None, :]  # (1,4)

    # output pointer offsets: (BLOCK_SIZE,4)
    out_offsets = (
        pid_x * 2048
        + pid_z * 256
        + tid[:, None] * 4
        + col_offsets
    )

    # ------------------------------------------------------------------
    # Loop over the 4 feature levels (i)
    # ------------------------------------------------------------------
    for i in range(4):
        # spatial shape of level i: (height, width)
        height = tl.load(value_spatial_shapes_ptr + i * 2).to(tl.int32)
        width = tl.load(value_spatial_shapes_ptr + i * 2 + 1).to(tl.int32)

        # start index of level i in the flattened `value` tensor
        level_start = tl.load(level_start_index_ptr + i).to(tl.int32)

        # common offset part for all corners of this level
        base_offset = level_start * 2048 + pid_z *256 +[:, None] * 4

        # --------------------------------------------------------------
        # Loop over the 4 sampling points per level (k)
        # --------------------------------------------------------------
        for k in range(4):
            # ------------------------------------------------------------------
            # Load normalized sampling location (x, y) – note the order matches CUDA
            # ------------------------------------------------------------------
            sample_idx = pid_x * 256 + pid_z * 32 + i * 8 + k * 2
            norm_x = tl.load(sampling_locations_ptr + sample_idx)          # original xy[1]
            norm_y = tl.load(sampling_locations_ptr + sample_idx + 1)      # original xy[0]

            # map to absolute grid coordinates
            xy_grid_y = norm_y * tl.cast(height, tl.float32) - 0.5
            xy_grid_x = norm_x * tl.cast(width, tl.float32) - 0.5

            # integer corner indices (floor and floor+1)
            row0 = tl.floor(xy_grid_y).to(tl.int32)   # floor(y)  -> xy_rounded[0]
            row1 = row0 + 1                           # xy_rounded[1]
            col0 = tl.floor(xy_grid_x).to(tl.int32)   # floor(x)  -> xy_rounded[2]
            col1 = col0 + 1                           # xy_rounded[3]

            # bilinear interpolation weights (scalar per block)
            row0_f = tl.cast(row0, tl.float32)
            row1_f = tl.cast(row1, tl.float32)
            col0_f = tl.cast(col0, tl.float32)
            col1_f = tl.cast(col1, tl.float32)

            w_tl = (row1_f - xy_grid_y) * (col1_f - xy_grid_x)   # top‑left
            w_tr = (xy_grid_y - row0_f) * (col1_f - xy_grid_x)   # top‑right
            w_bl = (row1_f - xy_grid_y) * (xy_grid_x - col0_f)   # bottom‑left
            w_br = (xy_grid_y - row0_f) * (xy_grid_x - col0_f)   # bottom‑right

            # linear indices of the four corners in the (height*width) plane
            idx00 = row0 * width + col0
            idx01 = row0 * width + col1
            idx10 = row1 * width + col0
            idx11 = row1 * width + col1

            offset00 = idx00 * 2048
            offset01 = idx01 * 2048
            offset10 = idx10 * 2048
            offset11 = idx11 * 2048

            # ------------------------------------------------------------------
            # Boundary masks for each corner
            # ------------------------------------------------------------------
            mask00 = (row0 >= 0) & (row0 < height) & (col0 >= 0) & (col0 < width)
            mask01 = (row0 >= 0) & (row0 < height) & (col1 >= 0) & (col1 < width)
            mask10 = (row1 >= 0) & (row1 < height) & (col0 >= 0) & (col0 < width)
            mask11 = (row1 >= 0) & (row1 < height) & (col1 >= 0) & (col1 < width)

            # ------------------------------------------------------------------
            # Load the 4‑channel corner values (shape: BLOCK_SIZE x 4)
            # ------------------------------------------------------------------
            corner00 = tl.load(
                value_ptr + base_offset + offset00 + col_offsets,
                mask=mask00,
                other=0.0,
            )
            corner01 = tl.load(
                value_ptr + base_offset + offset01 + col_offsets,
                mask=mask01,
                other=0.0,
            )
            corner10 = tl.load(
                value_ptr + base_offset + offset10 + col_offsets,
                mask=mask10,
                other=0.0,
            )
            corner11 = tl.load(
                value_ptr + base_offset + offset11 + col_offsets,
                mask=mask11,
                other=0.0,
            )

            # ------------------------------------------------------------------
            # Bilinear interpolation
            # ------------------------------------------------------------------
            interpolated = (
                corner00 * w_tl +
                corner01 * w_tr +
                corner10 * w_bl +
                corner11 * w_br
            )

            # ------------------------------------------------------------------
            # Load the scalar attention weight for this (i, k) pair
            # ------------------------------------------------------------------
            weight_idx = pid_x * 128 + pid_z * 16 + i * 4 + k
            att_weight = tl.load(attention_weights_ptr + weight_idx)

            # ------------------------------------------------------------------
            # Accumulate into the per‑thread sum
            # ------------------------------------------------------------------
            att_sum += interpolated * att_weight

    # ----------------------------------------------------------------------
    # Write the final 4‑channel result for each thread
    # ----------------------------------------------------------------------
    tl.store(output_ptr + out_offsets, att_sum)


# ----------------------------------------------------------------------
# Python wrapper (entry point) – matches the original CUDA kernel signature
# ----------------------------------------------------------------------
def torch_kernel(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """
    Triton implementation of the original CUDA kernel.
    Parameter order matches the external `cuda_kernel` function:
        (value, value_spatial_shapes, level_start_index,
         sampling_locations, attention_weights, output)
    """
    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------
    for name, tensor in {
        "value": value,
        "value_spatial_shapes": value_spatial_shapes,
        "level_start_index": level_start_index,
        "sampling_locations": sampling_locations,
        "attention_weights": attention_weights,
        "output": output,
    }.items():
        if not tensor.is_cuda:
            raise RuntimeError(f"{name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")

    # ------------------------------------------------------------------
    # Ensure correct dtypes (int32 for indices, float32 for data)
    # ------------------------------------------------------------------
    value_spatial_shapes = value_spatial_shapes.to(torch.int32).contiguous()
    level_start_index = level_start_index.to(torch.int32).contiguous()

    # ------------------------------------------------------------------
    # Kernel launch configuration – mirrors the original CUDA launch
    # ------------------------------------------------------------------
    BLOCK_SIZE = 64          # __launch_bounds__(64)
    lq = 100                 # number of query positions (grid dim x)
    n = 1                    # unused second grid dimension
    m = 8                    # number of heads (grid dim z)
    grid = (lq, n, m)       # (blockIdx.x, blockIdx.y, blockIdx.z)

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
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Optional synchronization for deterministic behavior
    torch.cuda.synchronize()