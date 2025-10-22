import math
import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: reproduces the exact indexing pattern of the original CUDA kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A,               # *float32, input tensor (NHWC layout)
    pool_max,        # *float32, output tensor (flattened)
    batch_size: tl.int32,
    channels: tl.int32,
    input_H: tl.int32,
    stride: tl.int32,
    output_H: tl.int32,
    # ------------------------------------------------------------------
    # Compile‑time constants derived from the launch configuration
    # ------------------------------------------------------------------
    BLOCK_SIZE: tl.constexpr,
    TILE_H: tl.constexpr,          # height of a tile (e.g., 4)
    TILE_W: tl.constexpr,          # width  of a tile (same as TILE_H for square tiles)
    TILES_PER_BATCH: tl.constexpr, # number of tiles per batch (output_H*output_H / (TILE_H*TILE_W))
    SHIFT_Y: tl.constexpr,         # bit shift to extract high bits of threadIdx.x for y computation
    LOG2_CHANNELS: tl.constexpr,   # log2(channels)
    X_MUL: tl.constexpr,           # TILE_H * TILE_W (block stride in x dimension)
    Y_DIV: tl.constexpr,           # output_H // TILE_H (division factor for y computation)
):
    # -------------------------------------------------
    # Thread / block indexing
    # -------------------------------------------------
    pid = tl.program_id(0).to(tl.int64)               # blockIdx.x
    tid = tl.arange(0, BLOCK_SIZE).to(tl.int64)       # threadIdx.x within the block

    # Linear output index for this thread
    offsets = pid * BLOCK_SIZE + tid

    # -------------------------------------------------
    # Decode (batch, channel, y_out, x_out) using the exact CUDA mapping
    # -------------------------------------------------
    batch_idx = pid // TILES_PER_BATCH
    tile_idx  = pid % TILES_PER_BATCH                     # B % 81 in the original code

    # channel = low  bits of threadIdx.x
    channel_idx = tid % tl.int64(channels)                # tid & (channels-1)

    # y_out = ((tile_idx * TILE_H + (tid >> SHIFT_Y)) // Y_DIV)
    y_out = ((tile_idx * TILE_H + (tid >> SHIFT_Y)) // Y_DIV)

    # x_out = ((pid * X_MUL + (tid >> LOG2_CHANNELS)) % output_H)
    x_out = ((pid * X_MUL + (tid >> LOG2_CHANNELS)) % output_H)

    # -------------------------------------------------
    # Compute input coordinates (top‑left corner of the pooling window)
    # -------------------------------------------------
    stride_i64   = tl.int64(stride)
    input_H_i64  = tl.int64(input_H)
    channels_i64 = tl.int64(channels)

    input_y = y_out * stride_i64
    input_x = x_out * stride_i64

    # Base offset for the element (batch, input_y, input_x, channel)
    # offset = ((batch * H + y) * W + x) * C + c
    input_offset_base = (
        ((batch_idx * input_H_i64 + input_y) * input_H_i64 + input_x) * channels_i64
        + channel_idx
    )

    # -------------------------------------------------
    # Mask for valid output threads
    # -------------------------------------------------
    total_output = batch_size * channels * output_H * output_H
    mask = offsets < tl.int64(total_output)

    # -------------------------------------------------
    # Max‑pool reduction over a 5×5 window (kernel_size is fixed to 5)
    # -------------------------------------------------
    max_val = tl.full([BLOCK_SIZE], -3.402823e+38, dtype=tl.float32)

    for rv0 in range(5):
        for rv1 in range(5):
            # Each step down a row adds input_H * channels,
            # each step right a column adds channels
            idx = input_offset_base + rv0 * input_H_i64 * channels_i64 + rv1 * channels_i64
            val = tl.load(A + idx, mask=mask, other=-3.402823e+38)
            max_val = tl.maximum(max_val, val)

    # -------------------------------------------------
    # Write the result
    # -------------------------------------------------
    tl.store(pool_max + offsets, max_val, mask=mask)


# ----------------------------------------------------------------------
# Wrapper that prepares launch parameters and launches the Triton kernel
# ----------------------------------------------------------------------
def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int,
):
    """
    Triton implementation of the original CUDA max‑pool kernel.
    Arguments must match the CUDA launch configuration.
    - input  : NHWC tensor of shape [batch, H, W, C] (float32, CUDA)
    - output : flat tensor of shape [batch * channels * output_H * output_H]
               (float32, CUDA)
    - batch_size, channels, input_H, kernel_size, stride : same as in CUDA
    """
    assert input.is_cuda and output.is_cuda, "Tensors must be CUDA tensors"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 supported"
    assert kernel_size == 5, "This Triton implementation only supports kernel_size=5"

    # Output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * channels * output_H * output_H

    # ------------------------------------------------------------------
    # Derive tiling constants (must match the CUDA kernel's arithmetic)
    # ------------------------------------------------------------------
    BLOCK_SIZE = 1024
    tile_area = BLOCK_SIZE // channels                     # = 16 for the test case
    tile_h = int(math.isqrt(tile_area))                   # = 4 (square tile)
    assert tile_h * tile_h == tile_area, "BLOCK_SIZE // channels must be a perfect square"
    tile_w = tile_h

    TILES_PER_BATCH = (output_H * output_H) // (tile_h * tile_w)  # = 81
    assert (output_H * output_H) % (tile_h * tile_w) == 0, "output_H*output_H must be divisible by tile area"

    log2_block = int(math.log2(BLOCK_SIZE))               # 10
    log2_tile_h = int(math.log2(tile_h))                  # 2
    SHIFT_Y = log2_block - log2_tile_h                    # 8  (matches T>>8 in CUDA)
    LOG2_CHANNELS = int(math.log2(channels))              # 6  (matches T>>6 in CUDA)
    X_MUL = tile_h * tile_w                               # 16 (matches B*16 in CUDA)
    Y_DIV = output_H // tile_h                            # 9  (matches division by 9 in CUDA)

    # Grid configuration
    num_blocks = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    _triton_kernel_impl[(num_blocks,)](
        input,
        output,
        batch_size,
        channels,
        input_H,
        stride,
        output_H,
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_H=tile_h,
        TILE_W=tile_w,
        TILES_PER_BATCH=TILES_PER_BATCH,
        SHIFT_Y=SHIFT_Y,
        LOG2_CHANNELS=LOG2_CHANNELS,
        X_MUL=X_MUL,
        Y_DIV=Y_DIV,
    )
    torch.cuda.synchronize()