import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    A,                     # pointer to input tensor (float32)
    pool_avg,              # pointer to output tensor (float32)
    batch_size: tl.int32,
    channels: tl.int32,
    input_H: tl.int32,
    kernel_size: tl.int32,
    stride: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    # Compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    total_output = batch_size * output_H * output_H * channels

    # Program ID (block index)
    pid = tl.program_id(0)
    # Offsets for each thread in this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask for threads that are within the valid range
    mask = offsets < total_output

    # Decode the linear offset into (batch, y_out, x_out, channel)
    c = offsets % channels
    tmp = offsets // channels
    x_out = tmp % output_H
    tmp = tmp // output_H
    y_out = tmp % output_H
    batch = tmp // output_H

    # Strides for the NHWC layout
    stride_H = input_H * channels   # distance between rows
    stride_W = channels             # distance between columns

    # Compute the top‑left corner of the pooling window in the input
    y_in = y_out * stride
    x_in = x_out * stride
    base = (
        batch * input_H * input_H * channels
        + y_in * stride_H
        + x_in * stride_W
        + c
    )

    # Accumulate the sum over the 5×5 window
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # Unrolled loops for kernel_size = 5
    for i in range(5):
        for j in range(5):
            idx = base + i * stride_H + j * stride_W
            sum_val += tl.load(A + idx, mask=mask, other=0.0)

    # Compute the average (1/25 = 0.04)
    avg = sum_val * (1.0 / (kernel_size * kernel_size))

    # Write the result
    tl.store(pool_avg + offsets, avg, mask=mask)


def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int
):
    """
    Entry point that mimics the original CUDA kernel signature.
    input  : NHWC tensor of shape (batch, input_H, input_H, channels)
    output : NHWC tensor of shape (batch, output_H, output_H, channels)
    """
    assert input.is_cuda and output.is_cuda, "Tensors must be on CUDA device"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Tensors must be float32"
    # Compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    # Total number of output elements
    output_size = batch_size * output_H * output_H * channels
    BLOCK_SIZE = 1024
    # Number of program instances (grid size)
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        output,
        batch_size,
        channels,
        input_H,
        kernel_size,
        stride,
        BLOCK_SIZE=BLOCK_SIZE
    )
    # Optional synchronization for debugging / timing
    # torch.cuda.synchronize()


# Example usage / simple test
if __name__ == "__main__":
    # Parameters matching the original CUDA kernel
    batch_size = 1
    channels = 64
    input_H = 112
    kernel_size = 5
    stride = 3

    # Create random input and allocate output
    input_tensor = torch.randn(batch_size, input_H, input_H, channels,
                               device="cuda", dtype=torch.float32)
    output_H = (input_H - kernel_size) // stride + 1
    output_tensor = torch.empty(batch_size, output_H, output_H, channels,
                                device="cuda", dtype=torch.float32)

    # Run Triton kernel
    triton_kernel(input_tensor, output_tensor,
                  batch_size, channels, input_H, kernel_size, stride)

    # Verify against PyTorch's avg_pool2d (expects NCHW)
    import torch.nn.functional as F
    input_nchw = input_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
    expected_nchw = F.avg_pool2d(input_nchw, kernel_size=kernel_size, stride=stride)
    expected = expected_nchw.permute(0, 2, 3, 1)   # NCHW -> NHWC

    # Check correctness
    if torch.allclose(output_tensor, expected, atol=1e-6):
        print("Triton kernel matches PyTorch avg_pool2d.")
    else:
        max_err = (output_tensor - expected).abs().max()
        print(f"Mismatch! max error = {max_err.item()}")