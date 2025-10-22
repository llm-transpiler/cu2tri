import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing 2D average pooling (NHWC layout)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # *float32, input tensor (NHWC)
    pool_avg_ptr,        # *float32, output tensor (NHWC)
    batch_size,          # int32
    channels,            # int32
    input_H,             # int32
    stride,              # int32
    output_H,            # int32
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)                     # block index (maps to a chunk of output)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # linear output indices
    output_size = batch_size * output_H * output_H * channels
    mask = offsets < output_size                # mask for out‑of‑bounds threads

    # --------------------------------------------------------------
    # Decode linear index to (b, y_out, x_out, c) for NHWC layout
    # --------------------------------------------------------------
    tmp = offsets
    c = tmp % channels
    tmp = tmp // channels
    x_out = tmp % output_H
    tmp = tmp // output_H
    y_out = tmp % output_H
    b = tmp // output_H

    # --------------------------------------------------------------
    # Accumulate sum over the K×K window
    # --------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for ky in range(KERNEL_SIZE):
        in_y = y_out * stride + ky
        for kx in range(KERNEL_SIZE):
            in_x = x_out * stride + kx
            # Input offset for NHWC layout:
            # ((b * H + in_y) * W + in_x) * C + c
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
    Triton implementation of 2D average pooling (valid padding) on NHWC tensors.
    The signature matches the original CUDA kernel.
    """
    assert input_tensor.is_cuda and output_tensor.is_cuda, "Tensors must be on CUDA device"
    assert input_tensor.dtype == torch.float32 and output_tensor.dtype == torch.float32, "Only float32 supported"
    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()

    # Compute output spatial dimension
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
        num_warps=4,
    )


# ----------------------------------------------------------------------
# Simple self‑test (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Example parameters (matching the failing test case)
    batch = 16
    channels = 64
    input_H = 112
    kernel = 5
    stride = 3

    output_H = (input_H - kernel) // stride + 1

    # Random input in NHWC layout
    inp = torch.randn(batch, input_H, input_H, channels, device="cuda", dtype=torch.float32)
    out = torch.empty(batch, output_H, output_H, channels, device="cuda", dtype=torch.float32)

    triton_kernel(inp, out, batch, channels, input_H, kernel, stride)

    # Reference using PyTorch (expects NCHW)
    inp_nchw = inp.permute(0, 3, 1, 2).contiguous()
    expected = torch.nn.functional.avg_pool2d(
        inp_nchw,
        kernel_size=kernel,
        stride=stride,
        padding=0,
        divisor_override=kernel * kernel,
    )
    # Convert Triton output back to NCHW for comparison
    out_nchw = out.permute(0, 3, 1, 2).contiguous()

    if torch.allclose(out_nchw, expected, atol=1e-2, rtol=1e-2):
        print("Success: Triton kernel matches PyTorch reference within tolerance.")
    else:
        max_err = (out_nchw - expected).abs().max()
        print(f"Mismatch: max absolute error = {max_err.item():.6f}")