import torch
import triton
import triton.language as tl

# Number of threads per program (matches the CUDA launch bounds)
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(
    A_ptr, out_ptr,
    batch_size, channels,
    input_H, input_W,
    output_H, output_W,
    stride,
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr
):
    """
    Triton kernel for 2‑D average pooling (NHWC layout).

    Computes:
        out[b, y, x, c] = mean(A[b, y*stride + ky, x*stride + kx, c])
    over a KERNEL_SIZE × KERNEL_SIZE window.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    idx = pid * BLOCK_SIZE + offs

    total_output = batch_size * output_H * output_W * channels
    mask = idx < total_output

    # Decode linear index -> (batch, y, x, channel)
    c = idx % channels
    tmp = idx // channels
    x = tmp % output_W
    tmp2 = tmp // output_W
    y = tmp2 % output_H
    b = tmp2 // output_H

    # Top‑left corner of the pooling window in the input tensor
    in_y = y * stride
    in_x = x * stride

    # Base offset for the input (NHWC layout)
    input_offset = ((b * input_H + in_y) * input_W + in_x) * channels + c

    # Accumulate sum over the KERNEL_SIZE × KERNEL_SIZE window
    sum_val = tl.float32(0.0)
    for ky in range(KERNEL_SIZE):
        for kx in range(KERNEL_SIZE):
            offset = input_offset + (ky * input_W + kx) * channels
            sum_val += tl.load(A_ptr + offset, mask=mask, other=0.0)

    # Compute the average
    avg = sum_val * (1.0 / (KERNEL_SIZE * KERNEL_SIZE))

    # Write the result
    tl.store(out_ptr + idx, avg, mask=mask)


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
    Entry point that mirrors the original CUDA kernel signature.
    Expects tensors in NHWC layout and dtype torch.float32.
    """
    assert input.is_cuda and output.is_cuda, "Tensors must reside on a CUDA device"
    assert input.is_contiguous() and output.is_contiguous(), "Tensors must be contiguous"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 is supported"

    # Output spatial dimensions (square assumption)
    output_H = (input_H - kernel_size) // stride + 1
    output_W = output_H

    total_output = batch_size * output_H * output_W * channels
    grid = ((total_output + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        output,
        batch_size,
        channels,
        input_H,
        input_H,          # input_W (square)
        output_H,
        output_W,
        stride,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_SIZE=kernel_size,
    )
    # torch.cuda.synchronize()  # Uncomment for debugging


# ----------------------------------------------------------------------
# Simple sanity‑check (run only when this file is executed directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example configuration matching the original CUDA launch bounds
    batch = 1
    channels = 64
    input_H = 112
    kernel = 5
    stride = 3

    # Derived output size
    output_H = (input_H - kernel) // stride + 1
    out_shape = (batch, output_H, output_H, channels)

    # Allocate input / output tensors
    x = torch.randn(batch, input_H, input_H, channels, device="cuda", dtype=torch.float32)
    y = torch.empty(out_shape, device="cuda", dtype=torch.float32)

    # Run Triton implementation
    triton_kernel(x, y, batch, channels, input_H, kernel, stride)

    # Reference using PyTorch (convert NHWC ↔ NCHW)
    x_nchw = x.permute(0, 3, 1, 2)  # NCHW
    y_ref_nchw = torch.nn.functional.avg_pool2d(x_nchw, kernel_size=kernel, stride=stride)
    y_ref = y_ref_nchw.permute(0, 2, 3, 1)  # back to NHWC

    # Verify correctness
    max_err = (y - y_ref).abs().max().item()
    print(f"Max absolute error vs. torch.nn.functional.avg_pool2d: {max_err:e}")