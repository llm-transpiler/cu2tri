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
    pid = tl.program_id(0)
    # Linear index of each output element processed by this thread
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE, dtype=tl.int64)
    total_output = batch_size * channels * output_H * output_H
    mask = offsets < total_output

    # Decompose linear index into (batch, channel, y_out, x_out)
    per_batch = channels * output_H * output_H
    batch_idx = offsets // per_batch
    rem = offsets % per_batch
    per_channel = output_H * output_H
    channel_idx = rem // per_channel
    rem2 = rem % per_channel
    y_out = rem2 // output_H
    x_out = rem2 % output_H

    # Compute base offset into the input tensor (NHWC)
    input_y = y_out * stride
    input_x = x_out * stride
    # ((batch * H + input_y) * W + input_x) * C + channel
    input_offset_base = ((batch_idx * input_H + input_y) * input_H + input_x) * channels + channel_idx

    # Initialize max with lowest float32 value
    max_val = tl.full([BLOCK_SIZE], -3.402823e+38, dtype=tl.float32)

    # Loop over the kernel window
    for rv0 in range(kernel_size):
        for rv1 in range(kernel_size):
            # Offset for this element: each row step adds input_H * channels,
            # each column step adds channels
            idx = input_offset_base + rv0 * input_H * channels + rv1 * channels
            val = tl.load(A + idx, mask=mask, other=-3.402823e+38)
            max_val = tl.maximum(max_val, val)

    # Write the result
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
    Triton implementation of a 2‑D max‑pool with kernel_size and stride.
    Mirrors the behavior of the original CUDA kernel.
    The input tensor must be in NHWC layout (shape: [batch, H, W, C]).
    The output tensor must be pre‑allocated with shape [batch * channels * output_H * output_H].
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
    # Ensure kernel completion before returning
    torch.cuda.synchronize()