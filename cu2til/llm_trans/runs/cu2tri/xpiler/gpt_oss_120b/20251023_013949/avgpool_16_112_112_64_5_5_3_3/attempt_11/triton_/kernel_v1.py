import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: average pooling (NHWC layout) with a square kernel.
# The kernel size is a compile‑time constant (KERNEL_SIZE) for optimal unrolling.
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A,                     # *float32, input tensor (NHWC)
    pool_avg,              # *float32, output tensor (NHWC)
    batch_size,            # int32
    channels,              # int32
    input_H,               # int32
    stride,                # int32
    output_H,              # int32
    output_size,           # int32
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    # Linear thread IDs within the grid
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < output_size

    # ------------------------------------------------------------------
    # Decode linear index -> (batch, out_y, out_x, channel)
    # ------------------------------------------------------------------
    out_spatial = output_H * output_H * channels
    n = offs // out_spatial
    rem = offs % out_spatial
    out_y = rem // (output_H * channels)
    rem2 = rem % (output_H * channels)
    out_x = rem2 // channels
    c = rem2 % channels

    # ------------------------------------------------------------------
    # Compute the top‑left corner of the KERNEL_SIZE×KERNEL_SIZE window
    # in the input tensor (NHWC layout)
    # ------------------------------------------------------------------
    in_y = out_y * stride
    in_x = out_x * stride
    # offset = ((n * input_H + in_y) * input_H + in_x) * channels + c
    base_offset = ((n * input_H + in_y) * input_H + in_x) * channels + c

    # ------------------------------------------------------------------
    # Accumulate the sum over the kernel window
    # ------------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for dy in range(KERNEL_SIZE):
        row_offset = base_offset + dy * input_H * channels
        for dx in range(KERNEL_SIZE):
            idx = row_offset + dx * channels
            val = tl.load(A + idx, mask=mask, other=0.0)
            sum_val += val

    # ------------------------------------------------------------------
    # Compute average and write back
    # ------------------------------------------------------------------
    avg = sum_val * (1.0 / (KERNEL_SIZE * KERNEL_SIZE))
    tl.store(pool_avg + offs, avg, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper matching the original CUDA kernel signature.
# ----------------------------------------------------------------------
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
    Triton implementation of the CUDA average‑pooling kernel.
    Expected tensor layout: NHWC (batch, height, width, channels).
    """
    assert input.is_cuda and output.is_cuda, "Input and output must be CUDA tensors"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 supported"

    # Compute output spatial dimension and total number of output elements
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    # Grid configuration (one program per BLOCK_SIZE elements)
    BLOCK_SIZE = 1024
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        output,
        batch_size,
        channels,
        input_H,
        stride,
        output_H,
        output_size,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_SIZE=kernel_size,
        num_warps=4,
    )