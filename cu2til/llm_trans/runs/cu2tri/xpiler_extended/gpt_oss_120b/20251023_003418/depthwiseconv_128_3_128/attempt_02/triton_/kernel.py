import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr, filter_ptr, output_ptr,
    input_height, input_width, output_height, output_width,
    kernel_size: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Output coordinates for this block
    x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    # Broadcast to a 2‑D grid (BLOCK_SIZE_X × BLOCK_SIZE_Y)
    x = x[:, None]          # shape (BLOCK_SIZE_X, 1)
    y = y[None, :]          # shape (1, BLOCK_SIZE_Y)

    # Mask for threads that lie inside the valid output region
    mask = (x < output_width) & (y < output_height)

    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_SIZE_X, BLOCK_SIZE_Y), dtype=tl.float32)

    # Depth‑wise convolution with a square kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Input coordinates corresponding to the current kernel element
            in_x = x + j
            in_y = y + i

            # Linear offset for the input element
            input_offset = ((in_y * input_width + in_x) * channels) + pid_c
            val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)

            # Linear offset for the filter element (scalar per channel)
            filter_offset = ((i * kernel_size + j) * channels) + pid_c
            f = tl.load(filter_ptr + filter_offset)

            # Accumulate the product
            acc += val * f

    # Write the accumulated result to the output tensor
    output_offset = ((y * output_width + x) * channels) + pid_c
    tl.store(output_ptr + output_offset, acc, mask=mask)

def triton_kernel(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_height: int,
    kernel_size: int,
    input_channels: int
):
    """
    Triton implementation of a depthwise 2‑D convolution.
    Parameters
    ----------
    input : torch.Tensor
        Flattened input tensor of shape (input_height * input_width * input_channels).
    kernel : torch.Tensor
        Flattened filter tensor of shape (kernel_size * kernel_size * input_channels).
    output : torch.Tensor
        Flattened output tensor of shape (output_height * output_width * input_channels).
    input_height : int
        Height (and width) of the square input feature map.
    kernel_size : int
        Spatial size of the square kernel (e.g., 3).
    input_channels : int
        Number of channels (depthwise convolution).
    """
    # Ensure tensors are on CUDA and contiguous
    assert input.is_cuda and kernel.is_cuda and output.is_cuda, "All tensors must be CUDA tensors"
    input = input.contiguous()
    kernel = kernel.contiguous()
    output = output.contiguous()

    # Compute output dimensions (square case)
    output_height = input_height - kernel_size + 1
    output_width = output_height

    # Block size tuned for the H800 80 GB SXM5
    BLOCK_SIZE_X = 32
    BLOCK_SIZE_Y = 32

    # Grid dimensions
    grid_x = (output_width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid_y = (output_height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    grid_z = input_channels
    grid = (grid_x, grid_y, grid_z)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        kernel,
        output,
        input_height,
        input_height,          # input_width (square)
        output_height,
        output_width,
        kernel_size=kernel_size,
        channels=input_channels,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )