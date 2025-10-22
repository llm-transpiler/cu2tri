import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    A_ptr,               # *float32, input in NHWC layout
    pool_avg_ptr,        # *float32, output in NHWC layout
    batch_size,          # int32
    channels,            # int32
    input_H,             # int32
    stride,              # int32
    output_H,            # int32
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_size = batch_size * output_H * output_H * channels
    mask = offsets < output_size

    # Decode linear index to (batch, y_out, x_out, channel) for NHWC layout
    c = offsets % channels
    tmp = offsets // channels
    x_out = tmp % output_H
    tmp2 = tmp // output_H
    y_out = tmp2 % output_H
    b = tmp2 // output_H

    # Accumulate sum over the kernel window
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for ky in range(KERNEL_SIZE):
        in_y = y_out * stride + ky
        for kx in range(KERNEL_SIZE):
            in_x = x_out * stride + kx
            # Flat index for input element in NHWC layout
            idx = ((b * input_H + in_y) * input_H + in_x) * channels + c
            val = tl.load(A_ptr + idx, mask=mask, other=0.0)
            sum_val += val

    scale = 1.0 / (KERNEL_SIZE * KERNEL_SIZE)
    avg = sum_val * scale
    tl.store(pool_avg_ptr + offsets, avg, mask=mask)


def triton_kernel(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int,
):
    """
    Triton implementation of 2D average pooling.
    Input layout : NHWC (batch, height, width, channels)
    Output layout: NHWC (batch, out_height, out_width, channels)
    """
    assert input_tensor.is_cuda and output_tensor.is_cuda, "Tensors must be on CUDA device"
    assert input_tensor.dtype == torch.float32 and output_tensor.dtype == torch.float32, "Only float32 tensors are supported"
    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()

    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    BLOCK_SIZE = 1024
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _triton_kernel_impl[grid](
        input_tensor,
        output_tensor,
        batch_size,
        channels,
        input_H,
        stride,
        output_H,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_SIZE=kernel_size,
        num_warps=32,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    # Example parameters
    batch = 2
    channels = 3
    input_H = 10
    kernel = 5
    stride = 1

    output_H = (input_H - kernel) // stride + 1

    # Input in NHWC layout
    input = torch.randn(batch, input_H, input_H, channels, device="cuda", dtype=torch.float32)
    output = torch.empty(batch, output_H, output_H, channels, device="cuda", dtype=torch.float32)

    triton_kernel(input, output, batch, channels, input_H, kernel, stride)

    # Verify against PyTorch's avg_pool2d (expects NCHW)
    input_nchw = input.permute(0, 3, 1, 2).contiguous()
    expected = torch.nn.functional.avg_pool2d(
        input_nchw,
        kernel_size=kernel,
        stride=stride,
        padding=0,
        divisor_override=kernel * kernel,
    )
    output_nchw = output.permute(0, 3, 1, 2).contiguous()
    if torch.allclose(output_nchw, expected, atol=1e-6):
        print("Triton kernel matches PyTorch reference.")
    else:
        max_err = (output_nchw - expected).abs().max()
        print(f"Mismatch! Max error: {max_err}")