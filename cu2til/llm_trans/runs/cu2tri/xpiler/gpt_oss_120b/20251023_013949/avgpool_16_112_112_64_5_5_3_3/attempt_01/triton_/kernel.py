import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    A_ptr,               # *float32
    pool_avg_ptr,        # *float32
    batch_size,          # int32
    channels,            # int32
    input_H,             # int32
    output_H,            # int32
    stride,              # int32
    BLOCK_SIZE: tl.constexpr,
    KSIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # linear thread ids within the grid
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE, dtype=tl.int64)

    total_outputs = batch_size * output_H * output_H * channels
    mask = offsets < total_outputs

    # decode NHWC coordinates from linear index
    c = offsets % channels
    tmp = offsets // channels
    w_out = tmp % output_H
    tmp = tmp // output_H
    h_out = tmp % output_H
    n = tmp // output_H

    # cast to int64 for arithmetic
    n = n.to(tl.int64)
    h_out = h_out.to(tl.int64)
    w_out = w_out.to(tl.int64)
    c = c.to(tl.int64)

    # accumulator for the sum of the kernel window
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # iterate over the kernel window
    for rv0 in tl.static_range(KSIZE):
        for rv1 in tl.static_range(KSIZE):
            in_h = h_out * stride + rv0
            in_w = w_out * stride + rv1
            # linear index into the input tensor (NHWC layout)
            input_offset = ((n * input_H + in_h) * input_H + in_w) * channels + c
            # load element (masked for out‑of‑range threads)
            val = tl.load(A_ptr + input_offset, mask=mask, other=0.0)
            sum_val += val

    # compute average
    scale = 1.0 / (KSIZE * KSIZE)
    avg = sum_val * scale

    # write result
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
    Triton implementation of a 2‑D average‑pooling kernel.
    The tensors are assumed to be in NHWC layout (batch, height, width, channels)
    and of type torch.float32 on the CUDA device.
    """
    # sanity checks
    assert input_tensor.is_cuda and output_tensor.is_cuda, "Tensors must be CUDA tensors"
    assert input_tensor.dtype == torch.float32 and output_tensor.dtype == torch.float32, "Only float32 supported"
    # ensure contiguous memory
    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()

    # compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    # total number of output elements
    output_size = batch_size * output_H * output_H * channels

    # launch configuration
    BLOCK_SIZE = 1024  # matches __launch_bounds__(1024) in the original CUDA kernel
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # launch the Triton kernel
    _triton_kernel_impl[grid](
        input_tensor,
        output_tensor,
        batch_size,
        channels,
        input_H,
        output_H,
        stride,
        BLOCK_SIZE=BLOCK_SIZE,
        KSIZE=kernel_size,
        num_warps=32,  # 1024 threads per block = 32 warps
    )