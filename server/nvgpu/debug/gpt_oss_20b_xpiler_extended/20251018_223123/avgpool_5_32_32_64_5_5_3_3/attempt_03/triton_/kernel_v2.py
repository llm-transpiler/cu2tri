import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr: tl.tensor,
    output_ptr: tl.tensor,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    input_H: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Constants
    blockdim = 1024

    # Compute block and thread indices
    program_id = tl.program_id(0)
    block_idx = program_id // blockdim
    thread_idx = program_id % blockdim

    # Condition used in the original CUDA kernel
    cond = (block_idx * 4 + (thread_idx >> 8)) < 125

    # Accumulate the 5x5 sum
    sum_val = 0.0
    for rv0 in range(5):
        for rv1 in range(5):
            if cond:
                # Compute the complex index expression
                base = (
                    ((block_idx * 4 + (thread_idx >> 8)) // 25) * 65536
                    + (((block_idx * 8 + (thread_idx >> 7)) % 50) // 5) * 6144
                    + ((block_idx * 6 + (thread_idx >> 6)) % 10) * 192
                )
                idx = base + rv0 * 2048 + rv1 * 64 + (thread_idx & 63)
                val = tl.load(input_ptr + idx)
                sum_val += val

    # Write the result if the condition holds
    if cond:
        out_idx = block_idx * 1024 + thread_idx
        tl.store(output_ptr + out_idx, sum_val * 0.04)


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
    Wrapper that launches the Triton kernel with the same semantics as the original CUDA kernel.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape (batch_size, channels, input_H, input_H) flattened to 1D.
    output : torch.Tensor
        Output tensor to be filled, shape (batch_size, channels, output_H, output_H) flattened to 1D.
    batch_size : int
        Number of batches.
    channels : int
        Number of channels.
    input_H : int
        Height (and width) of the input feature map.
    kernel_size : int
        Size of the pooling kernel.
    stride : int
        Stride of the pooling operation.
    """
    # Ensure tensors are on GPU and contiguous
    if not input.is_cuda or not output.is_cuda:
        raise ValueError("Input and output tensors must be CUDA tensors.")
    input = input.contiguous()
    output = output.contiguous()

    # Compute output height and total output size
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    # Launch the Triton kernel
    grid = (output_size,)
    _triton_kernel_impl[grid](
        input,
        output,
        batch_size,
        channels,
        input_H,
        kernel_size,
        stride,
        num_warps=4,
        num_stages=2,
    )