import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel_impl(
    A,                     # *float32, input tensor (NHWC)
    pool_avg,              # *float32, output tensor (NHWC)
    batch_size: tl.int32,  # runtime
    channels: tl.int32,
    input_H: tl.int32,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    output_H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # -------------------------------------------------------------------------
    # Thread / block identifiers
    # -------------------------------------------------------------------------
    block_id = tl.program_id(0)               # blockIdx.x (scalar)
    thread_id = tl.arange(0, BLOCK_SIZE)      # threadIdx.x (vector)
    pid = block_id * BLOCK_SIZE + thread_id    # linear output index (vector)

    # -------------------------------------------------------------------------
    # Guard out‑of‑bounds threads
    # -------------------------------------------------------------------------
    output_size = batch_size * channels * output_H * output_H
    mask = pid < output_size

    # -------------------------------------------------------------------------
    # Pre‑computed strides (NHWC layout)
    # -------------------------------------------------------------------------
    batch_stride = input_H * input_H * channels          # 802816 for 112×112×64
    row_stride = stride * input_H * channels             # 21504  (stride·H·C)
    kernel_row_stride = input_H * channels               # 7168   (H·C)
    col_stride = stride * channels                       # 192    (stride·C)
    channel_stride = channels                            # 64

    # -------------------------------------------------------------------------
    # Decode the original CUDA indexing scheme
    # -------------------------------------------------------------------------
    batch_offset = (block_id // 81) * batch_stride
    row_offset = (((block_id % 81) * 4 + (thread_id >> 8)) // 9) * row_stride
    col_offset = (((block_id * 16) + (thread_id >> 6)) % 36) * col_stride
    channel_offset = thread_id & 63

    base = batch_offset + row_offset + col_offset + channel_offset

    # -------------------------------------------------------------------------
    # Accumulate sum over the 5×5 kernel window
    # -------------------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for rv0 in range(kernel_size):
        for rv1 in range(kernel_size):
            offset = base + rv0 * kernel_row_stride + rv1 * channel_stride
            val = tl.load(A, offset=offset, mask=mask, other=0.0)
            sum_val = sum_val + val

    # -------------------------------------------------------------------------
    # Compute average and write result
    # -------------------------------------------------------------------------
    inv_kernel_area = 1.0 / (kernel_size * kernel_size)   # 1/25 = 0.04
    avg = sum_val * inv_kernel_area
    tl.store(pool_avg, offset=pid, value=avg, mask=mask)


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
    Triton implementation of the average‑pooling kernel.
    Mirrors the original CUDA kernel's signature and behavior.
    """
    # Ensure tensors are contiguous and reside on the same GPU device
    input = input.contiguous()
    output = output.contiguous()

    # Output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * channels * output_H * output_H
    BLOCK_SIZE = 1024

    # 1‑D grid configuration
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch kernel
    _triton_kernel_impl[grid](
        input,
        output,
        batch_size,
        channels,
        input_H,
        kernel_size=kernel_size,
        stride=stride,
        output_H=output_H,
        BLOCK_SIZE=BLOCK_SIZE
    )