import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: average pooling over a 5x5 window (generic kernel size)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A,                     # input pointer (float32)
    pool_avg,              # output pointer (float32)
    batch_size,            # int
    channels,              # int
    input_H,               # int (assumed square input: H = W)
    kernel_size: tl.constexpr,   # int (e.g., 5)
    stride: tl.constexpr,        # int (e.g., 1)
    BLOCK_SIZE: tl.constexpr     # int (must be 1024 to match launch bounds)
):
    pid = tl.program_id(0)                     # block index
    offs = tl.arange(0, BLOCK_SIZE)            # thread indices within the block
    gid = pid * BLOCK_SIZE + offs               # global output index

    # ------------------------------------------------------------------
    # Compute output dimensions (NHWC layout)
    # ------------------------------------------------------------------
    output_H = (input_H - kernel_size) // stride + 1
    output_W = output_H                         # square output
    total_output = batch_size * output_H * output_W * channels

    # Mask for out-of-bounds threads (when total_output is not a multiple of BLOCK_SIZE)
    mask = gid < total_output

    # ------------------------------------------------------------------
    # Decode (batch, out_y, out_x, channel) from linear index `gid`
    # ------------------------------------------------------------------
    c = gid % channels
    tmp = gid // channels
    out_x = tmp % output_W
    tmp = tmp // output_W
    out_y = tmp % output_H
    b = tmp // output_H

    # Top-left corner of the pooling window in the input tensor
    in_y = out_y * stride
    in_x = out_x * stride

    # Base offset for the input tensor (NHWC layout)
    # index = ((b * H + in_y) * W + in_x) * C + c
    base_offset = ((b * input_H + in_y) * input_H + in_x) * channels + c

    # ------------------------------------------------------------------
    # Accumulate sum over the kernel window
    # ------------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for rv0 in tl.static_range(kernel_size):
        for rv1 in tl.static_range(kernel_size):
            # Offset for the current element in the window
            offset = base_offset + (rv0 * input_H + rv1) * channels
            val = tl.load(A + offset, mask=mask, other=0.0)
            sum_val += val

    # ------------------------------------------------------------------
    # Compute average and write result
    # ------------------------------------------------------------------
    scale = 1.0 / (kernel_size * kernel_size)   # 1/25 for kernel_size=5
    avg = sum_val * scale
    tl.store(pool_avg + gid, avg, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function matching the original CUDA kernel signature
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
    Triton implementation of the average pooling kernel.
    Parameters match the original CUDA kernel signature exactly.
    """
    # ------------------------------------------------------------------
    # Basic sanity checks
    # ------------------------------------------------------------------
    assert input.is_cuda and output.is_cuda, "Input and output must be CUDA tensors"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Tensors must be float32"
    # Ensure contiguous memory layout
    input = input.contiguous()
    output = output.contiguous()

    # ------------------------------------------------------------------
    # Compute output size and launch configuration
    # ------------------------------------------------------------------
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    BLOCK_SIZE = 1024                     # matches __launch_bounds__(1024)
    grid = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # ------------------------------------------------------------------
    # Launch Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        input,
        output,
        batch_size,
        channels,
        input_H,
        kernel_size=kernel_size,   # constexpr
        stride=stride,             # constexpr
        BLOCK_SIZE=BLOCK_SIZE,     # constexpr
        num_warps=32               # 1024 threads = 32 warps
    )
    # Synchronize to make sure kernel has finished (optional)
    torch.cuda.synchronize()