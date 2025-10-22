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
    kernel_size: tl.constexpr,   # compile‑time constant (5 in our tests)
    stride: tl.int32,
    output_H: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # -------------------------------------------------
    # Thread and block indexing
    # -------------------------------------------------
    pid = tl.program_id(0).to(tl.int64)                     # block index
    tid = tl.arange(0, BLOCK_SIZE).to(tl.int64)             # thread index within block
    offsets = pid * BLOCK_SIZE + tid                         # global linear output index

    total_output = batch_size * channels * output_H * output_H
    mask = offsets < total_output

    # -------------------------------------------------
    # Decode linear index into (batch, channel, y_out, x_out)
    # -------------------------------------------------
    per_batch = channels * output_H * output_H
    batch_idx = offsets // per_batch
    rem = offsets % per_batch

    per_channel = output_H * output_H
    channel_idx = rem // per_channel
    rem2 = rem % per_channel

    y_out = rem2 // output_H
    x_out = rem2 % output_H

    # -------------------------------------------------
    # Convert to int64 for address arithmetic
    # -------------------------------------------------
    batch_idx_i64   = batch_idx
    channel_idx_i64 = channel_idx
    y_out_i64       = y_out
    x_out_i64       = x_out

    stride_i64   = stride.to(tl.int64)
    input_H_i64  = input_H.to(tl.int64)
    channels_i64 = channels.to(tl.int64)

    # Top‑left corner of the pooling window in the input tensor
    input_y = y_out_i64 * stride_i64
    input_x = x_out_i64 * stride_i64

    # Base offset for the (batch, input_y, input_x, channel) element
    input_offset_base = (
        ((batch_idx_i64 * input_H_i64 + input_y) * input_H_i64 + input_x) * channels_i64
        + channel_idx_i64
    )

    # -------------------------------------------------
    # Max‑pool reduction over the kernel window
    # -------------------------------------------------
    max_val = tl.full([BLOCK_SIZE], -3.402823e+38, dtype=tl.float32)

    for rv0 in tl.static_range(kernel_size):
        for rv1 in tl.static_range(kernel_size):
            idx = input_offset_base + rv0 * input_H_i64 * channels_i64 + rv1 * channels_i64
            val = tl.load(A + idx, mask=mask, other=-3.402823e+38)
            max_val = tl.maximum(max_val, val)

    # -------------------------------------------------
    # Write result
    # -------------------------------------------------
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
    Triton implementation of the original CUDA max‑pool kernel.
    Arguments:
        input  – NHWC tensor of shape [batch, H, W, C] (float32, CUDA)
        output – flat tensor of shape [batch * channels * output_H * output_H] (float32, CUDA)
        batch_size, channels, input_H, kernel_size, stride – same as in the CUDA launch
    """
    assert input.is_cuda and output.is_cuda, "T must be CUDA tensors"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 supported"

    # Compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * channels * output_H * output_H

    BLOCK_SIZE = 1024
    num_blocks = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel; kernel_size is a compile‑time constant (constexpr)
    _triton_kernel_impl[(num_blocks,)](
        input,
        output,
        batch_size,
        channels,
        input_H,
        kernel_size,          # constexpr argument
        stride,
        output_H,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.cuda.synchronize()