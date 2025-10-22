import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: one thread computes one output element (NCHW layout)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # *float32, input tensor (NCHW)
    pool_avg_ptr,        # *float32, output tensor (NCHW)
    batch_size,          # int32
    channels,            # int32
    input_H,             # int32
    stride,              # int32
    output_H,            # int32
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # linear output indices
    output_size = batch_size * channels * output_H * output_H
    mask = offsets < output_size                # out-of-bounds mask

    # --------------------------------------------------------------
    # Decode linear index to (b, c, y_out, x_out) in NCHW order
    # --------------------------------------------------------------
    tmp = offsets
    b = tmp // (channels * output_H * output_H)
    tmp = tmp % (channels * output_H * output_H)
    c = tmp // (output_H * output_H)
    tmp = tmp % (output_H * output_H)
    y_out = tmp // output_H
    x_out = tmp % output_H

    # --------------------------------------------------------------
    # Accumulate sum over the KxK window
    # --------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for ky in range(KERNEL_SIZE):
        in_y = y_out * stride + ky
        for kx in range(KERNEL_SIZE):
            in_x = x_out * stride + kx
            # flat index for input element in NCHW layout
            idx = ((b * channels + c) * input_H + in_y) * input_H + in_x
            val = tl.load(A_ptr + idx, mask=mask, other=0.0)
            sum_val += val

    # --------------------------------------------------------------
    # Compute average and write back
    # --------------------------------------------------------------
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
    Triton implementation of 2D average pooling (valid padding) on NCHW tensors.
    Mirrors the original CUDA kernel signature.
    """
    # ------------------------------------------------------------------
    # Sanity checks and layout preparation
    # ------------------------------------------------------------------
    assert input_tensor.is_cuda and output_tensor.is_cuda, "Tensors must reside on CUDA device"
    assert input_tensor.dtype == torch.float32 and output_tensor.dtype == torch.float32, "Only float32 supported"
    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()

    # ------------------------------------------------------------------
    # Compute output dimensions
    # ------------------------------------------------------------------
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * channels * output_H * output_H

    # ------------------------------------------------------------------
    # Launch configuration
    # ------------------------------------------------------------------
    BLOCK_SIZE = 1024
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
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
        num_warps=4,          # reasonable default for 1024‑thread blocks
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

    # Random input in NCHW layout
    inp = torch.randn(batch, channels, input_H, input_H, device="cuda", dtype=torch.float32)
    out = torch.empty(batch, channels, output_H, output_H, device="cuda", dtype=torch.float32)

    # Run Triton kernel
    triton_kernel(inp, out, batch, channels, input_H, kernel, stride)

    # Reference using PyTorch
    ref = torch.nn.functional.avg_pool2d(
        inp,
        kernel_size=kernel,
        stride=stride,
        padding=0,
        divisor_override=kernel * kernel,
    )

    # Verify correctness
    if torch.allclose(out, ref, atol=1e-2, rtol=1e-2):
        print("Success: Triton kernel matches PyTorch reference within tolerance.")
    else:
        max_abs_err = (out - ref).abs().max()
        max_rel_err = ((out - ref).abs() / (ref.abs() + 1e-12)).max()
        print(f"Mismatch! Max abs error = {max_abs_err.item():.6f}, Max rel error = {max_rel_err.item():.6f}")