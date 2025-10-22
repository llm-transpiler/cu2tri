import torch
import triton
import triton.language as tl

# Fixed launch configuration (matches the original CUDA launch bounds)
BLOCK_SIZE = 1024  # 1024 threads per block


@triton.jit
def _triton_kernel_impl(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_H,
    input_W,
    output_H,
    stride,
    TILE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that performs a 5×5 average‑pool with a configurable stride.
    Layout: NHWC (batch, height, width, channels).
    """
    # -------------------------------------------------------------
    # Compute tiling layout (each block processes a TILE_SIZE×TILE_SIZE tile)
    # -------------------------------------------------------------
    tiles_per_dim = (output_H + TILE_SIZE - 1) // TILE_SIZE
    tiles_per_batch = tiles_per_dim * tiles_per_dim

    pid = tl.program_id(0)                     # global block index

    batch = pid // tiles_per_batch              # which batch this block belongs to
    tile_idx = pid % tiles_per_batch            # linear tile index inside the batch
    tile_y = tile_idx // tiles_per_dim          # tile row
    tile_x = tile_idx % tiles_per_dim           # tile column

    out_y_base = tile_y * TILE_SIZE
    out_x_base = tile_x * TILE_SIZE

    # -------------------------------------------------------------
    # Thread‑level decomposition (channel + intra‑tile position)
    # -------------------------------------------------------------
    thread_idx = tl.arange(0, BLOCK_SIZE)      # [0, 1, ..., BLOCK_SIZE-1]

    channel = thread_idx % channels             # channel index (0 .. C‑1)
    sub = thread_idx // channels                # 0 .. TILE_SIZE*TILE_SIZE‑1
    sub_y = sub // TILE_SIZE
    sub_x = sub % TILE_SIZE

    out_y = out_y_base + sub_y                  # output row for this thread
    out_x = out_x_base + sub_x                  # output column for this thread

    # Guard against out‑of‑bounds (only needed when output_H is not a multiple of TILE_SIZE)
    mask = (out_y < output_H) & (out_x < output_H)

    # -------------------------------------------------------------
    # Compute the top‑left corner of the pooling window in the input
    # -------------------------------------------------------------
    in_y_base = out_y * stride
    in_x_base = out_x * stride

    # -------------------------------------------------------------
    # Accumulate the 5×5 sum
    # -------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    K = 5  # kernel size (fixed to 5)

    for i in range(K):
        for j in range(K):
            in_y = in_y_base + i
            in_x = in_x_base + j
            # Flat input index: ((batch * H + in_y) * W + in_x) * C + channel
            input_offset = ((batch * input_H + in_y) * input_W + in_x) * channels + channel
            sum_val += tl.load(input_ptr + input_offset, mask=mask, other=0.0)

    # -------------------------------------------------------------
    # Write the average (1/25 = 0.04)
    # -------------------------------------------------------------
    avg = sum_val * (1.0 / (K * K))

    # Flat output index: ((batch * output_H + out_y) * output_H + out_x) * C + channel
    output_offset = ((batch * output_H + out_y) * output_H + out_x) * channels + channel
    tl.store(output_ptr + output_offset, avg, mask=mask)


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
    Entry‑point wrapper that mimics the original CUDA kernel signature.
    Performs a 5×5 average‑pool with the given stride on NHWC tensors.
    """
    # -----------------------------------------------------------------
    # Basic sanity checks (mirrors expectations of the original CUDA code)
    # -----------------------------------------------------------------
    if not input.is_cuda or not output.is_cuda:
        raise RuntimeError("Input and output tensors must be CUDA tensors.")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("Only float32 tensors are supported.")
    if not input.is_contiguous() or not output.is_contiguous():
        raise RuntimeError("Input and output tensors must be contiguous.")
    if kernel_size != 5:
        raise ValueError("This Triton implementation only supports kernel_size == 5.")

    # Compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1

    # Expected tensor shapes (NHWC layout)
    expected_input_shape = (batch_size, input_H, input_H, channels)
    expected_output_shape = (batch_size, output_H, output_H, channels)

    if input.shape != expected_input_shape:
        raise RuntimeError(
            f"Input shape {input.shape} does not match expected {expected_input_shape}."
        )
    if output.shape != expected_output_shape:
        raise RuntimeError(
            f"Output shape {output.shape} does not match expected {expected_output_shape}."
        )

    # -----------------------------------------------------------------
    # Tiling configuration: each block processes a TILE_SIZE×TILE_SIZE tile
    # with all channels (BLOCK_SIZE = TILE_SIZE² * channels)
    # -----------------------------------------------------------------
    TILE_SIZE = 4  # 4×4 output elements per block → 4*4*64 = 1024 threads
    if TILE_SIZE * TILE_SIZE * channels != BLOCK_SIZE:
        raise RuntimeError(
            f"BLOCK_SIZE mismatch: TILE_SIZE² * channels = {TILE_SIZE * TILE_SIZE * channels}, "
            f"but BLOCK_SIZE is {BLOCK_SIZE}. This kernel assumes channels = 64."
        )

    tiles_per_dim = (output_H + TILE_SIZE - 1) // TILE_SIZE
    total_blocks = batch_size * tiles_per_dim * tiles_per_dim
    grid = (total_blocks,)

    # -----------------------------------------------------------------
    # Launch the Triton kernel
    # -----------------------------------------------------------------
    _triton_kernel_impl[grid](
        input,
        output,
        batch_size,
        channels,
        input_H,
        input_H,          # input_W (square input)
        output_H,
        stride,
        TILE_SIZE=TILE_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Ensure kernel completion before returning to Python
    torch.cuda.synchronize()