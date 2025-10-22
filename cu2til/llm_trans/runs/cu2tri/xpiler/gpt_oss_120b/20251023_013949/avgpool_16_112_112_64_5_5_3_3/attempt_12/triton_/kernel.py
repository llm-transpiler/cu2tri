import torch
import triton
import triton.language as tl

# Triton kernel implementing average pooling with a square kernel.
# kernel size and stride are compile‑time constants for optimal unrolling.
@triton.jit
def _triton_kernel_impl(
    A,                     # input tensor pointer (float32)
    pool_avg,              # output tensor pointer (float32)
    batch_size,            # int32
    channels,              # int32
    input_H,               # int32
    output_H,              # int32
    BLOCK_SIZE: tl.constexpr,   # threads per block (e.g., 1024)
    KERNEL_SIZE: tl.constexpr,  # pooling kernel size (e.g., 5)
    STRIDE: tl.constexpr        # stride (e.g., 1)
):
    pid = tl.program_id(0)

    # Global linear index for each thread in the grid
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    total_output = batch_size * channels * output_H * output_H
    mask = offsets < total_output

    # ------------------------------------------------------------------
    # Decode flat output index into (batch, channel, y, x) coordinates.
    # Layout: N (batch) → C (channel) → H (output_H) → W (output_H)
    # index = ((n * channels + c) * output_H + y) * output_H + x
    # ------------------------------------------------------------------
    tmp = offsets
    n = tmp // (channels * output_H * output_H)
    tmp = tmp % (channels * output_H * output_H)
    c = tmp // (output_H * output_H)
    tmp = tmp % (output_H * output_H)
    y = tmp // output_H
    x = tmp % output_H

    # Base coordinates in the input tensor (top‑left corner of the pooling window)
    in_y = y * STRIDE
    in_x = x * STRIDE

    # ------------------------------------------------------------------
    # Accumulate the sum over the KERNEL_SIZE × KERNEL_SIZE window
    # ------------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for ky in range(KERNEL_SIZE):
        for kx in range(KERNEL_SIZE):
            iy = in_y + ky
            ix = in_x + kx
            # Input index in NCHW layout:
            # ((n * channels + c) * input_H + iy) * input_H + ix
            input_idx = ((n * channels + c) * input_H + iy) * input_H + ix
            val = tl.load(A + input_idx, mask=mask, other=0.0)
            sum_val += val

    # Compute the average
    avg = sum_val * (1.0 / (KERNEL_SIZE * KERNEL_SIZE))

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
    Wrapper that mirrors the original CUDA kernel signature.
    `input` and `output` must be contiguous CUDA tensors of dtype torch.float32.
    """
    assert input.is_cuda and output.is_cuda, "Tensors must be on CUDA device"
    assert input.dtype == torch.float32 and output.dtype == torch.float32
    assert input.is_contiguous() and output.is_contiguous()

    # Compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1

    # Total number of output elements
    total_output = batch_size * channels * output_H * output_H

    # Block size (must match the constexpr used in the kernel)
    BLOCK_SIZE = 1024

    # Grid configuration (1‑D)
    grid = ((total_output + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        output,
        batch_size,
        channels,
        input_H,
        output_H,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_SIZE=kernel_size,
        STRIDE=stride,
        num_warps=32   # 32 warps → 1024 threads per block
    )