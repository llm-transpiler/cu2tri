import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    A_ptr,
    pool_avg_ptr,
    batch_size,
    channels,
    input_H,
    stride,
    output_H,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
):
    """
    Triton implementation of a 5×5 average pooling (stride=1) that works for
    arbitrary tensor layouts by using explicit strides.
    """
    pid = tl.program_id(0)
    # Linear thread IDs within the grid
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_output = batch_size * channels * output_H * output_H
    mask = offs < total_output

    # Convert to 64‑bit for safe arithmetic
    offs_i = offs.to(tl.int64)

    # Decompose linear index into (n, c, y, x) in row‑major logical order
    per_batch = channels * output_H * output_H
    n = offs_i // per_batch
    rem = offs_i % per_batch
    c = rem // (output_H * output_H)
    rem2 = rem % (output_H * output_H)
    out_y = rem2 // output_H
    out_x = rem2 % output_H

    # Accumulate the sum of the kernel window
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for rv0 in tl.static_range(KERNEL_SIZE):
        for rv1 in tl.static_range(KERNEL_SIZE):
            in_y = out_y * stride + rv0
            in_x = out_x * stride + rv1
            # Compute flat input offset using provided strides
            offset = (
                n * in_stride_n
                + c * in_stride_c
                + in_y * in_stride_h
                + in_x * in_stride_w
            )
            a = tl.load(A_ptr + offset, mask=mask, other=0.0)
            sum_val += a

    # Compute average
    avg = sum_val * (1.0 / (KERNEL_SIZE * KERNEL_SIZE))

    # Compute flat output offset using provided strides
    out_offset = (
        n * out_stride_n
        + c * out_stride_c
        + out_y * out_stride_h
        + out_x * out_stride_w
    )
    tl.store(pool_avg_ptr + out_offset, avg, mask=mask)


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
    Wrapper that launches the Triton kernel. Mirrors the original CUDA kernel
    signature exactly.
    """
    assert input.is_cuda and output.is_cuda, "input and output must be CUDA tensors"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 supported"
    if kernel_size != 5:
        raise NotImplementedError("Only kernel_size == 5 is supported in this Triton implementation")

    # Compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * channels * output_H * output_H

    # Strides (in elements) for input and output tensors
    in_stride_n, in_stride_c, in_stride_h, in_stride_w = input.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = output.stride()

    BLOCK_SIZE = 1024  # matches the CUDA launch configuration
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
        in_stride_n,
        in_stride_c,
        in_stride_h,
        in_stride_w,
        out_stride_n,
        out_stride_c,
        out_stride_h,
        out_stride_w,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_SIZE=kernel_size,
        num_warps=32,  # 1024 threads = 32 warps
    )