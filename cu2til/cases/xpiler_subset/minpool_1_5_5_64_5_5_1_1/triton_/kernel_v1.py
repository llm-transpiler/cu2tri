import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel_impl(
    A_ptr,
    pool_min_ptr,
    # Constants are hardcoded in the CUDA kernel.
    # We pass them as constexpr for clarity and performance.
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    INPUT_STRIDE_H: tl.constexpr,
    INPUT_STRIDE_W: tl.constexpr,
    NUM_CHANNELS: tl.constexpr,
):
    """
    Triton kernel to replicate the provided CUDA code.
    Each program instance in the grid simulates one CUDA block.
    The work of the first NUM_CHANNELS threads within a block is vectorized.
    """
    # channel_offsets corresponds to `threadIdx.x` in the range [0, 63].
    channel_offsets = tl.arange(0, NUM_CHANNELS)

    # Initialize a vector of accumulators, one for each "thread"/channel.
    # This corresponds to `float pool_min_local[1]; pool_min_local[0] = 3.402823e+38f;`
    min_vals = tl.full(shape=(NUM_CHANNELS,), value=float('inf'), dtype=tl.float32)

    # The nested loops correspond to `for (int rv0 = 0; rv0 < 5; ++rv0)` etc.
    for rv0 in range(KERNEL_H):
        for rv1 in range(KERNEL_W):
            # Calculate pointers to the input data for all channels simultaneously.
            # This corresponds to `(rv0 * 320) + (rv1 * 64) + threadIdx.x`
            input_offsets = rv0 * INPUT_STRIDE_H + rv1 * INPUT_STRIDE_W + channel_offsets
            
            # Load a vector of NUM_CHANNELS values from A.
            current_vals = tl.load(A_ptr + input_offsets)
            
            # Update the minimum values vector-wise.
            # This corresponds to `min(pool_min_local[0], A[...])`
            min_vals = tl.minimum(min_vals, current_vals)

    # Store the final results.
    # This corresponds to `pool_min[((int)threadIdx.x)] = pool_min_local[0];`
    # Since every program instance (block) does this, the race condition from the
    # original CUDA code is faithfully replicated.
    output_offsets = channel_offsets
    tl.store(pool_min_ptr + output_offsets, min_vals)


def triton_kernel(input: torch.Tensor, output: torch.Tensor, batch_size: int,
                  channels: int, input_H: int, kernel_size: int,
                  stride: int):
    """
    Wrapper function for the Triton kernel, with a signature and functionality
    identical to the original CUDA host function.

    Args:
        input (torch.Tensor): Input tensor.
        output (torch.Tensor): Output tensor.
        batch_size (int): Batch size.
        channels (int): Number of channels.
        input_H (int): Height of the input.
        kernel_size (int): Size of the pooling kernel.
        stride (int): Stride of the pooling operation.
    """
    # The original CUDA kernel is highly specialized with hardcoded values.
    # We add assertions to ensure the inputs match these assumptions.
    assert channels == 64, "This kernel is hardcoded for 64 channels."
    assert kernel_size == 5, "This kernel is hardcoded for a 5x5 kernel."
    assert input.dtype == torch.float32, "Input tensor must be float32."
    assert output.dtype == torch.float32, "Output tensor must be float32."
    assert input.is_cuda and output.is_cuda, "Tensors must be on a CUDA device."

    # Grid configuration logic from the original CUDA code.
    output_H = (input_H - kernel_size) // stride + 1
    # Assuming symmetric input and output (W=H) for grid calculation.
    output_W = output_H
    output_size = batch_size * output_H * output_W * channels

    # The CUDA code uses a fixed block size of 1024.
    cuda_block_size = 1024
    
    # The number of blocks is calculated to cover the entire conceptual output.
    num_blocks = (output_size + cuda_block_size - 1) // cuda_block_size

    # In our Triton kernel, each program instance corresponds to one CUDA block.
    # The grid is therefore 1D with size `num_blocks`.
    grid = (num_blocks,)

    # Launch the Triton kernel.
    # The hardcoded constants from the CUDA kernel are passed as constexpr.
    _triton_kernel_impl[grid](
        input,
        output,
        KERNEL_H=5,
        KERNEL_W=5,
        INPUT_STRIDE_H=320,
        INPUT_STRIDE_W=64,
        NUM_CHANNELS=64,
    )

# Example usage for verification
if __name__ == '__main__':
    # Parameters from the wrapper function signature
    batch_size = 1
    channels = 64
    input_H = 224 # A typical value, though it only affects grid size
    kernel_size = 5
    stride = 2

    # The CUDA kernel reads from a specific memory layout.
    # The indexing implies a contiguous block of memory representing a patch of
    # shape (H=5, W=5, C=64) with strides (320, 64, 1).
    # The largest offset is 4*320 + 4*64 + 63 = 1599.
    # The input tensor needs to be at least this large.
    input_tensor_size = 2048
    input_tensor = torch.randn(input_tensor_size, dtype=torch.float32, device='cuda')

    # Manually place known minimum values to verify correctness.
    # For channel 10, the minimum will be -100.0
    # Index for (rv0=2, rv1=3, channel=10) is 2*320 + 3*64 + 10 = 842
    input_tensor[842] = -100.0

    # For channel 33, the minimum will be -200.0
    # Index for (rv0=4, rv1=0, channel=33) is 4*320 + 0*64 + 33 = 1313
    input_tensor[1313] = -200.0

    # The output tensor only needs to be of size 64, as that's where the kernel writes.
    output_tensor = torch.zeros(channels, dtype=torch.float32, device='cuda')

    # Run the Triton kernel
    triton_kernel(input_tensor, output_tensor, batch_size, channels, input_H, kernel_size, stride)

    # --- Verification ---
    # Calculate the expected minimum for channel 10 on the CPU for comparison.
    min_val_ch10_cpu = float('inf')
    for rv0 in range(5):
        for rv1 in range(5):
            idx = rv0 * 320 + rv1 * 64 + 10
            min_val_ch10_cpu = min(min_val_ch10_cpu, input_tensor[idx].item())

    # Calculate the expected minimum for channel 33.
    min_val_ch33_cpu = float('inf')
    for rv0 in range(5):
        for rv1 in range(5):
            idx = rv0 * 320 + rv1 * 64 + 33
            min_val_ch33_cpu = min(min_val_ch33_cpu, input_tensor[idx].item())

    print("Triton kernel executed.")
    print(f"Output for channel 10: {output_tensor[10].item():.6f}")
    print(f"Expected minimum for channel 10: {min_val_ch10_cpu:.6f}")
    print(f"Output for channel 33: {output_tensor[33].item():.6f}")
    print(f"Expected minimum for channel 33: {min_val_ch33_cpu:.6f}")

    # Check if the results match
    assert torch.isclose(output_tensor[10], torch.tensor(min_val_ch10_cpu))
    assert torch.isclose(output_tensor[33], torch.tensor(min_val_ch33_cpu))
    print("\nVerification successful!")

    # Note on the race condition:
    output_H_calc = (input_H - kernel_size) // stride + 1
    output_size_calc = batch_size * output_H_calc * output_H_calc * channels
    num_blocks_calc = (output_size_calc + 1024 - 1) // 1024
    print(f"\nNote: The kernel was launched with a grid of {num_blocks_calc}, replicating the race condition.")
    print("In this specific case, the result is deterministic because all blocks compute the same value.")