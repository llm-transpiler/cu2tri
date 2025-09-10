import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel_impl(
    # Pointers to tensors
    A_ptr,
    pool_max_ptr,
    # Tensor dimensions
    INPUT_H,
    INPUT_W,
    CHANNELS,
    OUTPUT_H,
    OUTPUT_W,
    # Strides for memory access (NHWC layout)
    A_BATCH_STRIDE,
    A_H_STRIDE,
    A_W_STRIDE,
    POOL_MAX_BATCH_STRIDE,
    POOL_MAX_H_STRIDE,
    POOL_MAX_W_STRIDE,
    # Pooling parameters (as compile-time constants for performance)
    KERNEL_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    # Meta-parameters
    BLOCK_C: tl.constexpr,
):
    """
    Triton kernel for 2D Max Pooling.
    This kernel computes the max pooling operation over a 2D window for each channel.
    Each program instance handles a block of channels for a single output pixel (n, oh, ow).
    """
    # 1. Calculate the program IDs to determine the work item
    # This program is responsible for one output pixel (n, oh, ow) and a block of channels.
    # The grid is 2D: (N * OH * OW, ceil_div(C, BLOCK_C))
    pid_now = tl.program_id(0)  # ID for the spatial dimensions (N, OH, OW)
    pid_c_block = tl.program_id(1)  # ID for the channel block

    # 2. Decompose the spatial program ID into (n, oh, ow) coordinates
    NUM_OUTPUT_PIXELS_PER_BATCH = OUTPUT_H * OUTPUT_W
    n = pid_now // NUM_OUTPUT_PIXELS_PER_BATCH
    oh_ow_flat = pid_now % NUM_OUTPUT_PIXELS_PER_BATCH
    oh = oh_ow_flat // OUTPUT_W
    ow = oh_ow_flat % OUTPUT_W

    # 3. Calculate channel offsets for this program instance
    # Each program handles a block of BLOCK_C channels.
    c_offsets = pid_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < CHANNELS

    # 4. Initialize a vector of max values for the channel block
    # The initial value is the most negative number to ensure any input value becomes the new max.
    max_vals = tl.full(shape=(BLOCK_C,), fill_value=-float('inf'), dtype=tl.float32)

    # 5. Calculate the top-left corner of the pooling window in the input tensor
    start_h = oh * STRIDE
    start_w = ow * STRIDE

    # 6. Iterate over the pooling window (e.g., 5x5)
    # These loops are unrolled by the Triton compiler because KERNEL_SIZE is a tl.constexpr,
    # which is critical for performance.
    for kh in range(KERNEL_SIZE):
        for kw in range(KERNEL_SIZE):
            # Current position in the input tensor's H and W dimensions
            h = start_h + kh
            w = start_w + kw

            # Calculate pointer to the start of the current element in the input tensor
            # This points to the first channel of the pixel at (n, h, w).
            A_offset = (n * A_BATCH_STRIDE + h * A_H_STRIDE + w * A_W_STRIDE)
            A_ptrs = A_ptr + A_offset + c_offsets

            # Load a block of channels for the current (h, w) position.
            # Masking ensures we don't read out of bounds on the channel dimension.
            # If a channel is masked, 'other' provides a default value that won't affect the max reduction.
            input_vals = tl.load(A_ptrs, mask=c_mask, other=-float('inf'))

            # Update the running max values element-wise.
            max_vals = tl.maximum(max_vals, input_vals)

    # 7. Calculate output pointers
    pool_max_offset = (n * POOL_MAX_BATCH_STRIDE +
                       oh * POOL_MAX_H_STRIDE +
                       ow * POOL_MAX_W_STRIDE)
    output_ptrs = pool_max_ptr + pool_max_offset + c_offsets

    # 8. Store the final max values to the output tensor
    # Masking ensures we don't write out of bounds on the channel dimension.
    tl.store(output_ptrs, max_vals, mask=c_mask)


def triton_kernel(input: torch.Tensor, output: torch.Tensor, batch_size: int,
                  channels: int, input_H: int, kernel_size: int, stride: int):
    """
    Wrapper function for the Triton max pooling kernel.

    This function sets up the grid and launches the Triton kernel, providing an
    interface identical to the original CUDA wrapper. It assumes the input and
    output tensors are in NHWC (channels-last) format.

    Args:
        input (torch.Tensor): The input tensor of shape (N, H, W, C).
        output (torch.Tensor): The output tensor to store the results.
        batch_size (int): The batch size of the input tensor.
        channels (int): The number of channels in the input tensor.
        input_H (int): The height of the input tensor.
        kernel_size (int): The size of the pooling window.
        stride (int): The stride of the pooling operation.
    """
    # Assuming input_W = input_H as it's not provided in the function signature
    input_W = input_H

    # Basic validation for tensor shapes and memory layout
    # For best performance, tensors should be contiguous in channels-last format.
    assert input.shape == (batch_size, input_H, input_W, channels)
    assert input.is_contiguous(memory_format=torch.channels_last)
    assert input.dtype == torch.float32

    # Calculate output dimensions
    output_H = (input_H - kernel_size) // stride + 1
    output_W = (input_W - kernel_size) // stride + 1

    assert output.shape == (batch_size, output_H, output_W, channels)
    assert output.is_contiguous(memory_format=torch.channels_last)
    assert output.dtype == torch.float32

    # Define the block size for the channel dimension.
    # 64 is chosen based on the original CUDA kernel's logic (`threadIdx.x < 64`).
    # This is a tunable parameter.
    BLOCK_C = 64

    # Configure the grid for the kernel launch.
    # The grid is 2D:
    # - dim 0: Total number of output pixels across the entire batch (N * OH * OW).
    # - dim 1: Number of channel blocks needed to cover all channels.
    grid = lambda meta: (
        batch_size * output_H * output_W,
        triton.cdiv(channels, meta['BLOCK_C'])
    )

    # Get tensor strides for NHWC layout.
    # For a contiguous channels-last tensor, stride(3) for channels is 1.
    A_BATCH_STRIDE, A_H_STRIDE, A_W_STRIDE, _ = input.stride()
    POOL_MAX_BATCH_STRIDE, POOL_MAX_H_STRIDE, POOL_MAX_W_STRIDE, _ = output.stride()

    # Launch the Triton kernel.
    _triton_kernel_impl[grid](
        # Pointers
        input,
        output,
        # Dimensions
        input_H,
        input_W,
        channels,
        output_H,
        output_W,
        # Strides
        A_BATCH_STRIDE,
        A_H_STRIDE,
        A_W_STRIDE,
        POOL_MAX_BATCH_STRIDE,
        POOL_MAX_H_STRIDE,
        POOL_MAX_W_STRIDE,
        # Pooling parameters (passed as constexpr)
        KERNEL_SIZE=kernel_size,
        STRIDE=stride,
        # Meta-parameters
        BLOCK_C=BLOCK_C,
        num_warps=4,  # A good starting point for tuning
    )