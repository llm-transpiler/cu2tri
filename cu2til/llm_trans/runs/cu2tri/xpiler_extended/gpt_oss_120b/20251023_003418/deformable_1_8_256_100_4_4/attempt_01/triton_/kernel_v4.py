import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing the original CUDA kernel
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
    # ------------------------------------------------------------------
    # Compile‑time constants (match the original kernel)
    # ------------------------------------------------------------------
    NUM_LEVELS = 4          # number of feature levels (i)
    NUM_POINTS = 4          # number of sampling points per level (k)
    HEADS = 8               # number of attention heads (blockIdx.z)
    CHANNELS = 256          # per‑head channel dimension (d)
    CH_PER_THREAD = 4       # channels processed by each thread
    SPATIAL_STRIDE = HEADS * CHANNELS          # 2048
    HEAD_STRIDE = CHANNELS                     # 256
    THREAD_STRIDE = CH_PER_THREAD              # 4

    # ------------------------------------------------------------------
    # Program IDs (block indices)
    # ------------------------------------------------------------------
    pid_x = tl.program_id(0)  # blockIdx.x  (query index)
    pid_z = tl.program_id(2)  # blockIdx.z  (head index)

    # ------------------------------------------------------------------
    # Thread index within the block
    # ------------------------------------------------------------------
    tid = tl.arange(0, BLOCK_SIZE, dtype=tl.int64)          # (BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Offsets for the 4 channels each thread writes/reads
    # ------------------------------------------------------------------
    ch_offsets = tl.arange(0, CH_PER_THREAD, dtype=tl.int64)  # (4,)

    # ------------------------------------------------------------------
    # Output pointer offsets (flattened tensor of shape (lq, m, d))
    # ------------------------------------------------------------------
    out_base = pid_x * SPATIAL_STRIDE + pid_z * HEAD_STRIDE + tid * THREAD_STRIDE
    out_offsets = out_base[:, None] + ch_offsets[None, :]      # (BLOCK_SIZE, 4)

    # ------------------------------------------------------------------
    # Accumulator for the 4 output channels per thread
    # ------------------------------------------------------------------
    att_sum = tl.zeros((BLOCK_SIZE, CH_PER_THREAD), dtype=tl.float32)

    # ------------------------------------------------------------------
    # Loop over feature levels (i)
    # ------------------------------------------------------------------
    for i in range(NUM_LEVELS):
        # Load spatial shape of level i: (height, width)
        height = tl.load(value_spatial_shapes_ptr + i * 2).to(tl.int32)
        width = tl.load(value_spatial_shapes_ptr + i * 2 + 1).to(tl.int32)

        # Load start index of level i in the flattened `value` tensor
        level_start = tl.load(level_start_index_ptr + i).to(tl.int32)

        # Base offset for this level (excluding spatial index)
        level_base = (
            tl.cast(level_start, tl.int64) * SPATIAL_STRIDE
            + tl.cast(pid_z, tl.int64) * HEAD_STRIDE
            + tid * THREAD_STRIDE
        )  # (BLOCK_SIZE,)

        # ------------------------------------------------------------------
        # Loop over sampling points per level (k)
        # ------------------------------------------------------------------
        for k in range(NUM_POINTS):
            # ------------------------------------------------------------------
            # Load normalized sampling location (xy[1], xy[0])
            # ------------------------------------------------------------------
            sample_idx = (
                pid_x * (HEADS * NUM_LEVELS * NUM_POINTS * 2)
                + pid_z * (NUM_LEVELS * NUM_POINTS * 2)
                + i * (NUM_POINTS * 2)
                + k * 2
            )
            xy1 = tl.load(sampling_locations_ptr + sample_idx)          # xy[1]
            xy0 = tl.load(sampling_locations_ptr + sample_idx + 1)      # xy[0]

            # ------------------------------------------------------------------
            # Map to absolute grid coordinates
            # ------------------------------------------------------------------
            xy_grid0 = xy0 * tl.cast(height, tl.float32) - 0.5  # corresponds to xy_grid[0]
            xy_grid1 = xy1 * tl.cast(width, tl.float32) - 0.5   # corresponds to xy_grid[1]

            # ------------------------------------------------------------------
            # Integer corner indices (floor and floor+1)
            # ------------------------------------------------------------------
            row0 = tl.floor(xy_grid0).to(tl.int32)   # xy_rounded[0]
            row1 = row0 + 1                           # xy_rounded[1]
            col0 = tl.floor(xy_grid1).to(tl.int32)   # xy_rounded[2]
            col1 = col0 + 1                           # xy_rounded[3]

            # ------------------------------------------------------------------
            # Boundary masks for the four corners
            # ------------------------------------------------------------------
            mask00 = (row0 >= 0) & (row0 < height) & (col0 >= 0) & (col0 < width)
            mask01 = (row0 >= 0) & (row0 < height) & (col1 >= 0) & (col1 < width)
            mask10 = (row1 >= 0) & (row1 < height) & (col0 >= 0) & (col0 < width)
            mask11 = (row1 >= 0) & (row1 < height) & (col1 >= 0) & (col1 < width)

            # ------------------------------------------------------------------
            # Linear indices of the four corners in the (height*width) plane
            # ------------------------------------------------------------------
            idx00 = row0 * width + col0
            idx01 = row0 * width + col1
            idx10 = row1 * width + col0
            idx11 = row1 * width + col1

            # ------------------------------------------------------------------
            # Offsets for the four corners (each corner occupies SPATIAL_STRIDE elements)
            # ------------------------------------------------------------------
            offset00 = level_base + tl.cast(idx00, tl.int64) * SPATIAL_STRIDE
            offset01 = level_base + tl.cast(idx01, tl.int64) * SPATIAL_STRIDE
            offset10 = level_base + tl.cast(idx10, tl.int64) * SPATIAL_STRIDE
            offset11 = level_base + tl.cast(idx11, tl.int64) * SPATIAL_STRIDE

            # ------------------------------------------------------------------
            # Load the 4‑channel corner values (shape: BLOCK_SIZE x 4)
            # ------------------------------------------------------------------
            corner00 = tl.load(
                value_ptr + offset00[:, None] + ch_offsets[None, :],
                mask=mask00,
                other=0.0,
            )
            corner01 = tl.load(
                value_ptr + offset01[:, None] + ch_offsets[None, :],
                mask=mask01,
                other=0.0,
            )
            corner10 = tl.load(
                value_ptr + offset10[:, None] + ch_offsets[None, :],
                mask=mask10,
                other=0.0,
            )
            corner11 = tl.load(
                value_ptr + offset11[:, None] + ch_offsets[None, :],
                mask=mask11,
                other=0.0,
            )

            # ------------------------------------------------------------------
            # Bilinear interpolation weights (scalar per block)
            # ------------------------------------------------------------------
            row0_f = tl.cast(row0, tl.float32)
            col0_f = tl.cast(col0, tl.float32)
            row1_f = row0_f + 1.0
            col1_f = col0_f + 1.0

            w_tl = (row1_f - xy_grid0) * (col1_f - xy_grid1)   # top‑left
            w_tr = (xy_grid0 - row0_f) * (col1_f - xy_grid1)   # top‑right
            w_bl = (row1_f - xy_grid0) * (xy_grid1 - col0_f)   # bottom‑left
            w_br = (xy_grid0 - row0_f) * (xy_grid1 - col0_f)   # bottom‑right

            # ------------------------------------------------------------------
            # Bilinear interpolation
            # ------------------------------------------------------------------
            interpolated = (
                corner00 * w_tl
                + corner01 * w_tr
                + corner10 * w_bl
                + corner11 * w_br
            )

            # ------------------------------------------------------------------
            # Load the scalar attention weight for this (i, k) pair
            # ------------------------------------------------------------------
            att_idx = (
                pid_x * (HEADS * NUM_LEVELS * NUM_POINTS)
                + pid_z * (NUM_LEVELS * NUM_POINTS)
                + i * NUM_POINTS
                + k
            )
            att_weight = tl.load(attention_weights_ptr + att_idx)

            # ------------------------------------------------------------------
            # Accumulate into the per‑thread sum
            # ------------------------------------------------------------------
            att_sum += interpolated * att_weight

    # ----------------------------------------------------------------------
    # Write the final 4‑channel result for each thread
    # ----------------------------------------------------------------------
    tl.store(output_ptr + out_offsets, att_sum)


# ----------------------------------------------------------------------
# Public API – matches the original CUDA kernel signature
# ----------------------------------------------------------------------
def ms_deform_attn_forward(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """
    Triton implementation of Multi‑Scale Deformable Attention.
    Argument order matches the original CUDA kernel:
        (value, value_spatial_shapes, level_start_index,
         sampling_locations, attention_weights, output)
    The function writes the result directly into ``output``.
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
    # Ensure correct dtypes
    # ------------------------------------------------------------------
    value_spatial_shapes = value_spatial_shapes.to(torch.int32).contiguous()
    level_start_index = level_start_index.to(torch.int32).contiguous()

    # ------------------------------------------------------------------
    # Determine grid dimensions from the output tensor
    # ------------------------------------------------------------------
    if output.dim() != 3:
        raise RuntimeError(
            f"output must be a 3‑D tensor of shape (lq, m, d), got shape {output.shape}"
        )
    lq, m, d = output.shape
    if d != 256:
        raise RuntimeError(f"output channel dimension must be 256, got {d}")

    # Grid: (lq, 1, m)  – blockIdx.x = query, blockIdx.z = head
    grid = (lq, 1, m)

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    BLOCK_SIZE = 64  # matches __launch_bounds__(64) in the CUDA kernel
    _triton_kernel_impl[grid](
        attention_weights,
        output,
        sampling_locations,
        value,
        level_start_index,
        value_spatial_shapes,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.cuda.synchronize()


# ----------------------------------------------------------------------
# Compatibility wrapper used by the test harness (torch_/ref.py)
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
    Entry point matching the original ``cuda_kernel`` signature.
    Delegates to the Triton implementation.
    """
    ms_deform_attn_forward(
        value,
        value_spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        output,
    )


__all__ = [
    "ms_deform_attn_forward",
    "torch_kernel",
    "_triton_kernel_impl",
]