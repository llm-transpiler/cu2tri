import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel_impl(
    A_ptr,
    pool_avg_ptr,
    output_size: tl.int32,
    # Shape parameters
    batch_size: tl.int32,
    channels: tl.int32,
    input_H: tl.int32,
    output_H: tl.int32,
    # Pooling parameters
    kernel_size: tl.int32,
    stride: tl.int32,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for 2D sum pooling.

    This kernel implements a 2D pooling operation that calculates the sum over a sliding window.
    It is a generalized and corrected version of the provided CUDA kernel. The original CUDA
    kernel was hardcoded for a specific configuration (3x3 kernel, 2x2x64 output tile) and
    was not safe for grids larger than one block.

    This implementation handles arbitrary input sizes, batch sizes, and pooling parameters.
    It assumes a square input (Height = Width) and NHWC (channels-last) memory layout for
    optimal memory access patterns.

    Each program in the grid processes a block of `BLOCK_SIZE` output elements.
    """
    # 1. Determine which output elements this program is responsible for.
    pid = tl.program_id(axis=0)
    output_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offsets < output_size

    # 2. Decode the 1D output offsets into 4D (N, H, W, C) coordinates.
    # Assuming square images (Width = Height) and NHWC layout.
    output_W = output_H
    input_W = input_H

    # Strides for the output tensor (NHWC layout)
    stride_out_w = channels
    stride_out_h = output_W * stride_out_w
    stride_out_n = output_H * stride_out_h

    # Vectorized decoding of 1D offsets to 4D coordinates
    n = output_offsets // stride_out_n
    h_out_rem = output_offsets % stride_out_n
    h_out = h_out_rem // stride_out_h
    w_out_rem = h_out_rem % stride_out_h
    w_out = w_out_rem // stride_out_w
    c = w_out_rem % stride_out_w

    # 3. Calculate the top-left corner of the pooling window in the input tensor.
    h_start = h_out * stride
    w_start = w_out * stride

    # 4. Initialize an accumulator for the sum.
    pool_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # 5. Iterate over the pooling window and accumulate values from the input tensor.
    # Strides for the input tensor (NHWC layout)
    stride_in_w = channels
    stride_in_h = input_W * stride_in_w
    stride_in_n = input_H * stride_in_h

    # Triton 2.1+ supports dynamic loop bounds.
    for rv0 in range(kernel_size):
        for rv1 in range(kernel_size):
            # Calculate current coordinates within the input tensor
            h_in = h_start + rv0
            w_in = w_start + rv1

            # Calculate the 1D offset for the input tensor
            input_offsets = (n * stride_in_n +
                             h_in * stride_in_h +
                             w_in * stride_in_w +
                             c)

            # Load data from the input tensor. The mask ensures we only load for
            # valid output elements. The problem formulation guarantees that input
            # accesses are within bounds.
            val = tl.load(A_ptr + input_offsets, mask=output_mask, other=0.0)
            pool_sum += val

    # 6. Store the final sum into the output tensor.
    # The CUDA kernel variable `pool_avg` actually stores the sum, not the average.
    # We replicate this behavior.
    tl.store(pool_avg_ptr + output_offsets, pool_sum, mask=output_mask)


def triton_kernel(input: torch.Tensor, output: torch.Tensor, batch_size: int,
                  channels: int, input_H: int, kernel_size: int,
                  stride: int):
    """
    Wrapper function for the Triton sum pooling kernel.

    This function serves as the entry point, mirroring the functionality and
    signature of the original `cuda_kernel` C++ function. It handles grid
    configuration and launches the Triton kernel.

    Args:
        input (torch.Tensor): The input tensor. Assumed to be of shape
                              (batch_size, input_H, input_W, channels)
                              and in NHWC (channels-last) memory format.
        output (torch.Tensor): The pre-allocated output tensor. Will be populated
                               with the pooling results.
        batch_size (int): The batch size of the input tensor.
        channels (int): The number of channels in the input tensor.
        input_H (int): The height of the input tensor (assuming square).
        kernel_size (int): The size of the pooling window.
        stride (int): The stride of the pooling operation.
    """
    # Basic validation
    assert input.is_cuda and output.is_cuda, "Tensors must be on a CUDA device."
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Tensors must be of type float32."
    # The kernel is optimized for NHWC (channels-last) format.
    # While not strictly enforced here, performance will be best if the input
    # tensor is contiguous in this format.

    # Calculate output dimensions
    output_H = (input_H - kernel_size) // stride + 1
    output_W = output_H  # Assuming square
    output_size = batch_size * output_H * output_W * channels

    assert output.numel() >= output_size, f"Output tensor is too small. Needs {output_size} elements, but has {output.numel()}."

    # Configure the launch grid
    # Each program will handle a block of output elements.
    BLOCK_SIZE = 1024  # Match the __launch_bounds__ from the CUDA kernel
    grid = (triton.cdiv(output_size, BLOCK_SIZE),)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        output,
        output_size,
        batch_size,
        channels,
        input_H,
        output_H,
        kernel_size,
        stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )


if __name__ == '__main__':
    # --- Verification and Performance Test ---

    # Define test parameters based on the CUDA kernel's implicit values
    batch_size = 4
    channels = 64
    input_H = 224  # A more realistic input size
    kernel_size = 3
    stride = 2

    # Derived values
    output_H = (input_H - kernel_size) // stride + 1
    input_W = input_H
    output_W = output_H

    # Create input and output tensors on the GPU
    # Use channels-last memory format for which the kernel is optimized
    input_tensor = torch.randn(
        batch_size, input_H, input_W, channels,
        device='cuda', dtype=torch.float32
    ).contiguous(memory_format=torch.channels_last)

    # The output tensor is written to as a flattened array, so we create it as such
    output_size = batch_size * output_H * output_W * channels
    triton_output = torch.empty(output_size, device='cuda', dtype=torch.float32)

    print("--- Correctness Check ---")
    print(f"Input shape (NHWC): ({batch_size}, {input_H}, {input_W}, {channels})")
    print(f"Output shape (NHWC): ({batch_size}, {output_H}, {output_W}, {channels})")

    # Run the Triton kernel
    triton_kernel(input_tensor, triton_output, batch_size, channels, input_H, kernel_size, stride)

    # Reshape Triton's flat output to 4D for comparison
    triton_output_reshaped = triton_output.reshape(batch_size, output_H, output_W, channels)

    # --- PyTorch Reference Implementation ---
    # PyTorch's pooling ops work on NCHW format, so we need to permute the input
    input_nchw = input_tensor.permute(0, 3, 1, 2).contiguous()

    # The CUDA/Triton kernel performs sum pooling. We can achieve this in PyTorch
    # by using AvgPool2d and multiplying by the number of elements in the window.
    pool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0)
    ref_output_nchw = pool(input_nchw) * (kernel_size * kernel_size)

    # Permute the reference output back to NHWC for comparison
    ref_output_nhwc = ref_output_nchw.permute(0, 2, 3, 1).contiguous()

    # Compare the results
    is_correct = torch.allclose(triton_output_reshaped, ref_output_nhwc, atol=1e-4, rtol=1e-4)
    print(f"Correctness test passed: {is_correct}")
    if not is_correct:
        print("Mismatch detected!")
        print("Triton output sample:", triton_output_reshaped.flatten()[:10])
        print("PyTorch ref sample:", ref_output_nhwc.flatten()[:10])
        
    assert is_correct

    # --- Performance Benchmark ---
    print("\n--- Performance Benchmark ---")
    
    # Use Triton's built-in benchmarking utility
    @triton.testing.perf_report(
        [triton.testing.Benchmark(
            x_names=['output_size'],
            x_vals=[128*128*c for c in [32, 64, 128]],
            line_arg='provider',
            line_vals=['triton', 'pytorch'],
            line_names=['Triton', 'PyTorch'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='ms',
            plot_name='sum-pool-performance',
            args={'batch_size': 2, 'input_H': 256, 'kernel_size': 3, 'stride': 2}
        )]
    )
    def benchmark(batch_size, input_H, channels, kernel_size, stride, provider, output_size):
        input_W = input_H
        output_H = (input_H - kernel_size) // stride + 1
        output_W = output_H
        
        input_t = torch.randn(
            batch_size, input_H, input_W, channels, device='cuda', dtype=torch.float32
        ).contiguous(memory_format=torch.channels_last)
        output_t = torch.empty(
            batch_size * output_H * output_W * channels, device='cuda', dtype=torch.float32
        )

        quantiles = [0.5, 0.2, 0.8]
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_kernel(input_t, output_t, batch_size, channels, input_H, kernel_size, stride),
                quantiles=quantiles
            )
        if provider == 'pytorch':
            input_nchw_bench = input_t.permute(0, 3, 1, 2).contiguous()
            pool_op = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: pool_op(input_nchw_bench) * (kernel_size * kernel_size),
                quantiles=quantiles
            )
        return ms, min_ms, max_ms

    # Run the benchmark. The results will be printed to the console.
    # To save a plot, uncomment the line below.
    # benchmark.run(show_plots=True, print_data=True, save_path='.')
    benchmark.run(show_plots=False, print_data=True)