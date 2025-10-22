import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024  # 1024 threads per block (matches original CUDA launch)


@triton.jit
def _triton_kernel_impl(
    A,               # *float32 input tensor (contiguous NCHW)
    pool_max,        # *float32 output tensor (contiguous NCHW)
    batch_size: tl.int32,
    channels: tl.int32,
    input_H: tl.int32,
    stride: tl.int32,
    output_H: tl.int32,
    KERNEL_SIZE: tl.constexpr,   # compile‑time kernel size (e.g. 5)
    BLOCK_SIZE: tl.constexpr,    # compile‑time block size (1024)
):
    pid = tl.program_id(0)                     # block index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # linear thread ids

    total_output = batch_size * channels * output_H * output_H
    mask = offs < total_output

    # ------------------------------------------------------------------
    # Guard out‑of‑bounds threads
    # ------------------------------------------------------------------
    if mask:
        # Decode flat output index into (batch, channel, y_out, x_out) for NCHW
        x_out = offs % output_H
        tmp = offs // output_H
        y_out = tmp % output_H
        tmp = tmp // output_H
        c = tmp % channels
        b = tmp // channels

        # Top‑left input coordinate of the pooling window
        y_in = y_out * stride
        x_in = x_out * stride

        # Base address of the top‑left element (flattened NCHW)
        base = ((b * channels + c) * input_H + y_in) * input_H + x_in

        # Initialise max value with -FLT_MAX
        max_val = tl.full([], -3.402823e+38, dtype=tl.float32)

        # 5×5 (or KERNEL_SIZE×KERNEL_SIZE) max‑pool reduction
        for rv0 in tl.static_range(KERNEL_SIZE):
            for rv1 in tl.static_range(KERNEL_SIZE):
                idx = base + rv0 * input_H + rv1
                val = tl.load(A + idx)
                max_val = tl.maximum(max_val, val)

        # Write result
        tl.store(pool_max + offs, max_val)


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
    Wrapper that mirrors the original CUDA kernel signature.
    Arguments:
        input      – CUDA tensor of shape (batch, channels, input_H, input_H)
        output     – CUDA tensor of shape (batch, channels, output_H, output_H)
        batch_size – number of batches (must match input.shape[0])
        channels   – number of channels (must match input.shape[1])
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
    if input.shape != (batch_size, channels, input_H, input_H):
        raise RuntimeError(
            f"input shape {input.shape} does not match expected "
            f"({batch_size}, {channels}, {input_H}, {input_H})"
        )

    # Compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    if output.shape != (batch_size, channels, output_H, output_H):
        raise RuntimeError(
            f"output shape {output.shape} does not match expected "
            f"({batch_size}, {channels}, {output_H}, {output_H})"
        )

    # Ensure contiguous layout
    if not input.is_contiguous():
        input = input.contiguous()
    if not output.is_contiguous():
        output = output.contiguous()

    # ------------------------------------------------------------------
    # Grid configuration
    # ------------------------------------------------------------------
    total_output = batch_size * channels * output_H * output_H
    grid = ((total_output + BLOCK_SIZE - 1) // BLOCK_SIZE,)

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
        num_warps=32,  # 1024 threads → 32 warps
    )
    # Synchronise to make the kernel effect visible to the caller
    torch.cuda.synchronize()