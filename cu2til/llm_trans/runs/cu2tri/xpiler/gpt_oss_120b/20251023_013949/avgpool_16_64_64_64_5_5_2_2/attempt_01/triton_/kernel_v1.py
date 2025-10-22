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
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
):
    """
    Triton kernel that computes a KERNEL_SIZE×KERNEL_SIZE average pooling.
    Each program instance processes BLOCK_SIZE output elements.
    """
    pid = tl.program_id(0)
    # Linear indices of the output elements handled by this program
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_output = batch_size * channels * output_H * output_H
    mask = offs < total_output

    # Use 64‑bit arithmetic for offsets
    offs_i = offs.to(tl.int64)

    # Decompose linear index into (n, c, y, x)
    per_batch = channels * output_H * output_H
    n = offs_i // per_batch
    rem = offs_i % per_batch
    c = rem // (output_H * output_H)
    rem2 = rem % (output_H * output_H)
    out_y = rem2 // output_H
    out_x = rem2 % output_H

    # Accumulate sum over the kernel window
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for rv0 in tl.static_range(KERNEL_SIZE):
        for rv1 in tl.static_range(KERNEL_SIZE):
            in_y = out_y * stride + rv0
            in_x = out_x * stride + rv1
            # Flattened input offset: ((n*C + c)*H + in_y)*W + in_x
            offset = ((n * channels + c) * input_H + in_y) * input_H + in_x
            a = tl.load(A_ptr + offset, mask=mask, other=0.0)
            sum_val += a

    avg = sum_val * (1.0 / (KERNEL_SIZE * KERNEL_SIZE))
    tl.store(pool_avg_ptr + offs_i, avg, mask=mask)

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
    Entry point mirroring the original CUDA kernel signature.
    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape (batch_size, channels, input_H, input_H), dtype=torch.float32, on CUDA.
    output : torch.Tensor
        Output tensor of shape (batch_size, channels, output_H, output_H), dtype=torch.float32, on CUDA.
    batch_size, channels, input_H, kernel_size, stride : int
        Pooling configuration. This implementation currently supports kernel_size == 5.
    """
    assert input.is_cuda and output.is_cuda, "input and output must be CUDA tensors"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 supported"
    if kernel_size != 5:
        raise NotImplementedError("Only kernel_size == 5 is supported in this Triton implementation")
    # Compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * channels * output_H * output_H

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
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_SIZE=kernel_size,
        num_warps=32,   # 1024 threads = 32 warps
    )