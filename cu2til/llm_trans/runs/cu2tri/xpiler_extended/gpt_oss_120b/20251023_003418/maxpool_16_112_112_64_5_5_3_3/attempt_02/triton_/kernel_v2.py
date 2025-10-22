import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    A,               # *float32, input tensor (NHWC layout)
    pool_max,        # *float32, output tensor (flattened)
    batch_size: tl.int32,
    channels: tl.int32,
    input_H: tl.int32,
    kernel_size: tl.int32,
    stride: tl.int32,
    output_H: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Program (block) ID
    pid = tl.program_id(0).to(tl.int64)

    # Linear offsets for this block (vector of size BLOCK_SIZE)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)

    # Total number of output elements
    total_output = tl.int64(batch_size * channels * output_H * output_H)

    # Mask for valid threads (those that map to a real output element)
    mask = offsets < total_output

    # ------------------------------------------------------------------
    # Decode the linear offset into (batch, channel, y_out, x_out)
    # ------------------------------------------------------------------
    per_batch = tl.int64(channels * output_H * output_H)
    batch_idx = offsets // per_batch
    rem = offsets % per_batch

    per_channel = tl.int64(output_H * output_H)
    channel_idx = rem // per_channel
    rem2 = rem % per_channel

    out_H_i64 = tl.int64(output_H)
    y_out = rem2 // out_H_i64
    x_out = rem2 % out_H_i64

    # ------------------------------------------------------------------
    # Compute the base offset into the input tensor (NHWC layout)
    # index = ((batch * H + y) * W + x) * C + c
    # ------------------------------------------------------------------
    stride_i64 = tl.int64(stride)
    input_y = y_out * stride_i64
    input_x = x_out * stride_i64

    # Cast everything to int64 for pointer arithmetic
    batch_idx = batch_idx.to(tl.int64)
    channel_idx = channel_idx.to(tl.int64)
    input_y = input_y.to(tl.int64)
    input_x = input_x.to(tl.int64)
    input_H_i64 = tl.int64(input_H)
    channels_i64 = tl.int64(channels)

    input_offset_base = (
        batch_idx * input_H_i64 * input_H_i64 * channels_i64
        + input_y * input_H_i64 * channels_i64
        + input_x * channels_i64
        + channel_idx
    )

    # ------------------------------------------------------------------
    # Perform the max‑pool reduction over a 5×5 window
    # (kernel_size is fixed to 5 in the original CUDA kernel)
    # ------------------------------------------------------------------
    max_val = tl.full((BLOCK_SIZE,), -3.402823e+38, dtype=tl.float32)

    for rv0 in range(5):
        for rv1 in range(5):
            idx = (
                input_offset_base
                + tl.int64(rv0) * input_H_i64 * channels_i64
                + tl.int64(rv1) * channels_i64
            )
            val = tl.load(A + idx, mask=mask, other=-3.402823e+38)
            max_val = tl.maximum(max_val, val)

    # ------------------------------------------------------------------
    # Write the result
    # ------------------------------------------------------------------
    tl.store(pool_max + offsets, max_val, mask=mask)


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
    Triton implementation of a 2‑D max‑pool with kernel size 5 and arbitrary stride.
    Input tensor must be in NHWC layout (shape: [batch, H, W, C]).
    Output tensor must be a flat tensor of shape [batch * channels * output_H * output_H].
    """
    assert input.is_cuda and output.is_cuda, "Tensors must be CUDA tensors"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 supported"

    # Compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * channels * output_H * output_H

    BLOCK_SIZE = 1024
    num_blocks = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[(num_blocks,)](
        input,
        output,
        batch_size,
        channels,
        input_H,
        kernel_size,
        stride,
        output_H,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.cuda.synchronize()