import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing 5×5 (or arbitrary) max‑pooling.
# The tensor layout is (batch, height, width, channels) for both input
# and output (channels are the innermost dimension).
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024  # matches the original CUDA launch configuration


@triton.jit
def _triton_kernel_impl(
    A,               # *float32  input tensor
    pool_max,        # *float32  output tensor
    batch_size,      # int32
    channels,        # int32
    input_H,         # int32  spatial size of the input (assumed square)
    stride,          # int32
    output_H,        # int32  spatial size of the output (assumed square)
    KERNEL_SIZE: tl.constexpr,  # compile‑time kernel dimension (e.g. 5)
    BLOCK_SIZE: tl.constexpr,   # compile‑time block size (1024)
):
    pid = tl.program_id(0)                     # block index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # linear thread ids

    # ------------------------------------------------------------------
    # Bounds check
    # ------------------------------------------------------------------
    total_output = batch_size * output_H * output_H * channels
    mask = offs < total_output

    # ------------------------------------------------------------------
    # Decode flat output index into (batch, y_out, x_out, channel)
    # ------------------------------------------------------------------
    c = offs % channels
    tmp = offs // channels
    x_out = tmp % output_H
    tmp = tmp // output_H
    y_out = tmp % output_H
    b = tmp // output_H

    # ------------------------------------------------------------------
    # Compute the base address of the top‑left element of the pooling window
    # Input layout: ((b * input_H + y) * input_H + x) * channels + c
    # ------------------------------------------------------------------
    y_in = y_out * stride
    x_in = x_out * stride
    base = ((b * input_H + y_in) * input_H + x_in) * channels + c

    # ------------------------------------------------------------------
    # Reduce over the KERNEL_SIZE × KERNEL_SIZE window
    # ------------------------------------------------------------------
    max_val = tl.full([BLOCK_SIZE], -3.402823e+38, dtype=tl.float32)  # -FLT_MAX
    for rv0 in tl.static_range(KERNEL_SIZE):
        for rv1 in tl.static_range(KERNEL_SIZE):
            # Offset for the current element inside the window
            idx = base + rv0 * input_H * channels + rv1 * channels
            val = tl.load(A + idx, mask=mask, other=-3.402823e+38)
            max_val = tl.maximum(max_val, val)

    # ------------------------------------------------------------------
    # Write the result
    # ------------------------------------------------------------------
    tl.store(pool_max + offs, max_val, mask=mask)


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
    Entry‑point that mirrors the original CUDA wrapper.
    Arguments:
        input      – CUDA tensor of shape (batch, input_H, input_H, channels)
        output     – CUDA tensor of shape (batch, output_H, output_H, channels)
        batch_size – number of batches
        channels   – number of channels (must match the innermost dimension)
        input_H    – spatial size of the input (assumed square)
        kernel_size– size of the pooling window (e.g. 5)
        stride     – stride of the pooling operation
    """
    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------
    if not input.is_cuda or not output.is_cuda:
        raise RuntimeError("input and output tensors must be CUDA tensors")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("input and output tensors must be torch.float32")
    if not input.is_contiguous():
        input = input.contiguous()
    if not output.is_contiguous():
        output = output.contiguous()

    # ------------------------------------------------------------------
    # Compute output spatial dimension
    # ------------------------------------------------------------------
    output_H = (input_H - kernel_size) // stride + 1
    if output_H <= 0:
        raise ValueError("Invalid output dimension computed")

    # ------------------------------------------------------------------
    # Grid configuration
    # ------------------------------------------------------------------
    output_size = batch_size * output_H * output_H * channels
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        input,
        output,
        batch_size,
        channels,
        input_H,
        stride,
        output_H,
        KERNEL_SIZE=kernel_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,  # 8 warps → 256 threads per warp group; adjust for your GPU
    )
    # Ensure completion before returning to the caller
    torch.cuda.synchronize()