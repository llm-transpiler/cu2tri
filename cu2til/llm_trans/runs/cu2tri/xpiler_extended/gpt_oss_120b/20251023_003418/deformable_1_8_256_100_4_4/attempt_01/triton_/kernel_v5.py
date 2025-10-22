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
    # Compile‑time constants
    NUM_LEVELS = 4          # number of feature levels (i)
    NUM_POINTS = 4          # number of sampling points per level (k)
    HEADS = 8               # number of attention heads (blockIdx.z)
    CHANNELS = 256          # per‑head channel dimension (d)
    CH_PER_THREAD = 4       # channels processed by each thread
    SPATIAL_STRIDE = HEADS * CHANNELS   # 2048
    HEAD_STRIDE = CHANNELS              # 256
    THREAD_STRIDE = CH_PER_THREAD       # 4

    # Program IDs (block indices)
    pid_x = tl.program_id(0)  # blockIdx.x  (query index)
    pid_z = tl.program_id(2)  # blockIdx.z  (head index)

    # Thread index within the block (0 .. BLOCK_SIZE‑1)
    tid = tl.arange(0, BLOCK_SIZE, dtype=tl.int64)          # (BLOCK_SIZE,)
    ch_offsets = tl.arange(0, CH_PER_THREAD, dtype=tl.int64)  # (4,)

    # Output offsets (flattened tensor of shape (lq, m, d))
    out_base = pid_x * SPATIAL_STRIDE + pid_z * HEAD_STRIDE + tid * THREAD_STRIDE
    out_offsets = out_base[:, None] + ch_offsets[None, :]    # (BLOCK_SIZE, 4)

    # Accumulator for the 4 output channels per thread
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
            # Index into sampling_locations (original order: xy[1] then xy[0])
            sample_idx = (
                pid_x * (HEADS * NUM_LEVELS * NUM_POINTS * 2)
                + pid_z * (NUM_LEVELS * NUM_POINTS * 2)
                + i * (NUM_POINTS * 2)
                + k * 2
            )
            xy1 = tl.load(sampling_locations_ptr + sample_idx)          # xy[1]
            xy0 = tl.load(sampling_locations_ptr + sample_idx + 1)      # xy[0]

            # Map to absolute grid coordinates
            xy_grid0 = xy0 * tl.cast(height, tl.float32) - 0.5  # vertical (height) coordinate
            xy_grid1 = xy1 * tl.cast(width, tl.float32) - 0.5   # horizontal (width) coordinate

            # Integer corner indices (floor and floor+1)
            row0 = tl.floor(xy_grid0).to(tl.int32)   # xy_rounded[0]
            row1 = row0 + 1                           # xy_rounded[1]
            col0 = tl.floor(xy_grid1).to(tl.int32)   # xy_rounded[2]
            col1 = col0 + 1                           # xy_rounded[3]

            # Boundary masks for each corner
            mask00 = (row0 >= 0) & (row0 < height) & (col0 >= 0) & (col0 < width)
            mask01 = (row0 >= 0) & (row0 < height) & (col1 >= 0) & (col1 < width)
            mask10 = (row1 >= 0) & (row1 < height) & (col0 >= 0) & (col0 < width)
            mask11 = (row1 >= 0) & (row1 < height) & (col1 >= 0) & (col1 < width)

            # Linear indices of the four corners in the (height*width) plane
            idx00 = row0 * width + col0
            idx01 = row0 * width + col1
            idx10 = row1 * width + col0
            idx11 = row1 * width + col1

            # Offsets for the four corners (each corner occupies SPATIAL_STRIDE elements)
            offset00 = level_base + tl.cast(idx00, tl.int64) * SPATIAL_STRIDE
            offset01 = level_base + tl.cast(idx01, tl.int64) * SPATIAL_STRIDE
            offset10 = level_base + tl.cast(idx10, tl.int64) * SPATIAL_STRIDE
            offset11 = level_base + tl.cast(idx11, tl.int64) * SPATIAL_STRIDE

            # Load the 4‑channel corner values (shape: BLOCK_SIZE x 4)
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

            # Bilinear interpolation weights (scalar per block)
            row0_f = tl.cast(row0, tl.float32)
            col0_f = tl.cast(col0, tl.float32)
            row1_f = row0_f + 1.0
            col1_f = col0_f + 1.0

            w_tl = (row1_f - xy_grid0) * (col1_f - xy_grid1)   # top‑left
            w_tr = (xy_grid0 - row0_f) * (col1_f - xy_grid1)   # top‑right
            w_bl = (row1_f - xy_grid0) * (xy_grid1 - col0_f)   # bottom‑left
            w_br = (xy_grid0 - row0_f) * (xy_grid1 - col0_f)   # bottom‑right

            # Bilinear interpolation
            interpolated = (
                corner00 * w_tl
                + corner01 * w_tr
                + corner10 * w_bl
                + corner11 * w_br
            )

            # Load the scalar attention weight for this (i, k) pair
            att_idx = (
                pid_x * (HEADS * NUM_LEVELS * NUM_POINTS)
                + pid_z * (NUM_LEVELS * NUM_POINTS)
                + i * NUM_POINTS
                + k
            )
            att_weight = tl.load(attention_weights_ptr + att_idx)

            # Accumulate into the per‑thread sum
            att_sum += interpolated * att_weight

    # Write the final 4‑channel result for each thread
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
    The result is written directly into ``output``.
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
    # Ensure correct dtypes for index tensors
    # ------------------------------------------------------------------
    value_spatial_shapes = value_spatial_shapes.to(torch.int32).contiguous()
    level_start_index = level_start_index.to(torch.int32).contiguous()

    # ------------------------------------------------------------------
    # Infer grid dimensions from the input tensors
    # ------------------------------------------------------------------
    # sampling_locations shape: (lq, m, num_levels * num_points * 2)
    lq = sampling_locations.shape[0]
    m = sampling_locations.shape[1]

    BLOCK_SIZE = 64  # matches __launch_bounds__(64) in the CUDA kernel
    grid = (lq, 1, m)  # (blockIdx.x, blockIdx.y, blockIdx.z)

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
    torch.cuda.synchronize()


def torch_kernel(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """
    Entry point used by the test harness (torch_/ref.py).
    Simply forwards to ``ms_deform_attn_forward``.
    """
    ms_deform_attn_forward(
        value,
        value_spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        output,
    )


# ----------------------------------------------------------------------
# Autograd Function (optional, for completeness)
# ----------------------------------------------------------------------
class MultiScaleDeformableAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, level_start_index,
                sampling_locations, attention_weights):
        # Allocate output tensor (shape: (lq, m, d))
        lq = sampling_locations.shape[0]
        m = sampling_locations.shape[1]
        d = value.shape[-1]  # expected to be 256
        output = torch.empty((lq, m, d), dtype=value.dtype, device=value.device)
        ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            output,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError(
            "Backward pass is not implemented for MultiScaleDeformableAttentionFunction"
        )


# ----------------------------------------------------------------------
# nn.Module wrapper (mirrors the original API)
# ----------------------------------------------------------------------
class MultiScaleDeformableAttention(torch.nn.Module):
    """
    nn.Module wrapper that provides a familiar PyTorch interface.
    """
    def __init__(self):
        super().__init__()

    def forward(self, value, value_spatial_shapes, level_start_index,
                sampling_locations, attention_weights):
        return MultiScaleDeformableAttentionFunction.apply(
            value,
            value_spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )


# Exported symbols
__all__ = [
    "torch_kernel",
    "ms_deform_attn_forward",
    "MultiScaleDeformableAttentionFunction",
    "MultiScaleDeformableAttention",
    "_triton_kernel_impl",
]