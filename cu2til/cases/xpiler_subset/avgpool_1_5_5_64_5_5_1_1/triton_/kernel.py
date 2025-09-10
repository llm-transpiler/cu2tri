import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel_impl(
    A_ptr,
    pool_avg_ptr,
    # Strides for navigating the input tensor (in number of elements)
    input_stride_n,
    input_stride_h,
    input_stride_w,
    # Dimensions for decoding the 1D program ID
    output_H,
    output_W,
    # Pooling parameters
    kernel_stride,
    # --- Constants derived from the original CUDA kernel ---
    # The number of channels processed in parallel by one program instance.
    BLOCK_C: tl.constexpr,
    # The hardcoded stride for the outer loop (rv0), corresponding to the height dimension.
    PATCH_STRIDE_H: tl.constexpr,
    # The hardcoded stride for the inner loop (rv1), corresponding to the width dimension.
    PATCH_STRIDE_W: tl.constexpr,
):
    """
    Triton kernel for a specific 5x5 average pooling operation.

    This kernel is a direct and corrected translation of the provided CUDA kernel.
    The original CUDA code contained a race condition due to missing global indexing
    and used hardcoded values (320, 64) for memory access, implying a very
    specific input tensor shape.

    This Triton kernel fixes the race condition by mapping each program instance
    to a unique output spatial location. It preserves the exact memory access
    pattern from the CUDA code by using the same hardcoded strides, making it a
    faithful conversion of the intended algorithm.

    The grid is 1D, where each program instance computes the result for one
    spatial location (n, h_out, w_out) across all 64 channels.
    """
    # Each program instance computes one output spatial location (n, h_out, w_out)
    # for all C channels. The grid is 1D over N * H_out * W_out.
    spatial_pid = tl.program_id(0)

    # Decode the 1D program ID into 3D coordinates (n, h_out, w_out).
    # This determines which output pixel this program is responsible for.
    w_out = spatial_pid % output_W
    h_out = (spatial_pid // output_W) % output_H
    n = spatial_pid // (output_H * output_W)

    # Calculate the base pointer to the top-left corner of the corresponding
    # input patch. This corresponds to input[n, h_out * stride, w_out * stride, 0].
    h_in_start = h_out * kernel_stride
    w_in_start = w_out * kernel_stride
    A_base_ptr = A_ptr + n * input_stride_n + h_in_start * input_stride_h + w_in_start * input_stride_w

    # Calculate the base pointer for the output vector.
    # This corresponds to output[n, h_out, w_out, 0].
    output_base_ptr = pool_avg_ptr + spatial_pid * BLOCK_C

    # Create a vector of offsets for the channel dimension [0, 1, ..., 63].
    c_offsets = tl.arange(0, BLOCK_C)

    # Initialize an accumulator vector for the sum, one element per channel.
    pool_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over the 5x5 window. The Triton compiler will unroll these loops.
    for rv0 in range(5):
        for rv1 in range(5):
            # Calculate offsets for loading from the input tensor.
            # This uses the specific strides (320, 64) from the original CUDA kernel,
            # preserving its unique memory access pattern.
            load_offsets = (rv0 * PATCH_STRIDE_H) + (rv1 * PATCH_STRIDE_W)

            # Load a vector of BLOCK_C channels simultaneously.
            vals = tl.load(A_base_ptr + load_offsets + c_offsets)

            # Accumulate the values vector-wise.
            pool_sum += vals

    # Final calculation: multiply by 1/25 (0.04) to get the average.
    pool_avg = pool_sum * 0.04

    # Store the resulting vector of BLOCK_C channel averages to the output tensor.
    tl.store(output_base_ptr + c_offsets, pool_avg)


def triton_kernel(input: torch.Tensor, output: torch.Tensor, batch_size: int,
                  channels: int, input_H: int, kernel_size: int, stride: int):
    """
    Wrapper function for the Triton average pooling kernel.

    This function provides an entry point with a signature identical to the original
    CUDA wrapper. It validates input parameters, calculates tensor strides,
    configures the launch grid, and executes the Triton kernel.

    The implementation assumes the input and output tensors are in NHWC (channels-last)
    memory format, which is common for performance-oriented computer vision tasks.

    Args:
        input (torch.Tensor): The input tensor, assumed to be in NHWC layout.
        output (torch.Tensor): The output tensor, where the results will be written.
        batch_size (int): The batch size of the input tensor.
        channels (int): The number of channels in the input tensor.
        input_H (int): The height of the input tensor (assuming square, H=W).
        kernel_size (int): The size of the pooling window.
        stride (int): The stride of the pooling operation.
    """
    # The original CUDA kernel has hardcoded values that imply specific tensor
    # dimensions. We add assertions here to ensure compatibility.
    if kernel_size != 5:
        raise ValueError(
            f"This kernel is specialized for kernel_size=5, but got {kernel_size}")
    if channels != 64:
        raise ValueError(
            f"This kernel is specialized for channels=64, but got {channels}")

    # The hardcoded strides (320, 64) in the CUDA kernel imply a relationship
    # between the input width and the number of channels for an NHWC layout:
    # Intra-patch stride_W = channels = 64
    # Intra-patch stride_H = input_W * channels = 320 => input_W = 320 / 64 = 5
    # We assume a square input, so input_H must also be 5.
    input_W = input_H
    if input_W != 5:
        raise ValueError(
            f"The CUDA kernel's memory access pattern is only correct for an input width of 5, but got {input_W}")

    # Ensure tensors are on the correct device and have a compatible memory layout.
    assert input.is_cuda and output.is_cuda, "Input and output tensors must be on a CUDA device."
    assert input.is_contiguous(memory_format=torch.channels_last), "Input tensor must be in NHWC format (channels-last)."
    assert output.is_contiguous(memory_format=torch.channels_last), "Output tensor must be in NHWC format (channels-last)."

    # Calculate output dimensions.
    output_H = (input_H - kernel_size) // stride + 1
    output_W = (input_W - kernel_size) // stride + 1

    # Get tensor strides for NHWC memory layout (in number of elements).
    input_stride_n = input.stride(0)
    input_stride_h = input.stride(1)
    input_stride_w = input.stride(2)

    # Configure the launch grid: Launch one program for each spatial location
    # in the output tensor, across the entire batch.
    num_output_pixels = batch_size * output_H * output_W
    grid = (num_output_pixels, )

    # Launch the Triton kernel.
    _triton_kernel_impl[grid](
        input,
        output,
        input_stride_n,
        input_stride_h,
        input_stride_w,
        output_H,
        output_W,
        stride,
        BLOCK_C=64,          # Corresponds to threadIdx.x range and channel count
        PATCH_STRIDE_H=320,  # Hardcoded value from CUDA kernel's rv0 loop
        PATCH_STRIDE_W=64,   # Hardcoded value from CUDA kernel's rv1 loop
    )