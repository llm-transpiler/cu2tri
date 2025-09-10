import torch
import triton
import triton.language as tl

# The actual Triton kernel implementation
@triton.jit
def _triton_kernel_impl(
    input_ptr,
    kernel_ptr,
    output_ptr,
    # Using tl.constexpr for dimensions allows Triton to generate more specialized and efficient code.
    # These values are derived directly from the CUDA code's logic and loops.
    BATCH_SIZE: tl.constexpr = 16,
    IN_C: tl.constexpr = 64,
    IN_H: tl.constexpr = 8,
    IN_W: tl.constexpr = 8,
    OUT_C: tl.constexpr = 128,
    OUT_H: tl.constexpr = 4,
    OUT_W: tl.constexpr = 4,
    K_H: tl.constexpr = 2,
    K_W: tl.constexpr = 2,
    STRIDE: tl.constexpr = 2,
):
    """
    Triton kernel for 2D convolution, mirroring the provided CUDA implementation.
    Each program instance computes a single 4x4 output feature map for one
    output channel and one batch item. This corresponds to the work of one CUDA thread block.
    """
    # --- Grid and Block Mapping ---
    # The CUDA grid is (output_channels, 1, batch_size).
    # We map this to a 2D Triton grid (output_channels, batch_size).
    # blockIdx.x -> oc
    oc = tl.program_id(0)
    # blockIdx.z -> bs
    bs = tl.program_id(1)

    # The CUDA thread block is (output_width, output_height), i.e., (4, 4).
    # We create a 4x4 block of offsets to compute all pixels in the output
    # feature map in parallel within this program instance.
    # threadIdx.y -> oh
    offs_oh = tl.arange(0, OUT_H)
    # threadIdx.x -> ow
    offs_ow = tl.arange(0, OUT_W)

    # Create 2D blocks for vectorized computation.
    # oh will have shape [4, 1], ow will have shape [1, 4].
    oh = offs_oh[:, None]
    ow = offs_ow[None, :]

    # --- Accumulator Initialization ---
    # This holds the 'sum' for each of the 16 output pixels.
    accumulator = tl.zeros((OUT_H, OUT_W), dtype=tl.float32)

    # --- Main Computation Loops ---
    # These loops iterate over input channels and the kernel dimensions,
    # exactly like the CUDA kernel.
    for ic in range(0, IN_C):
        for kh in range(0, K_H):
            for kw in range(0, K_W):
                # --- Input Index Calculation ---
                # Calculate the input height and width for each output pixel.
                ih = oh * STRIDE + kh
                iw = ow * STRIDE + kw

                # Calculate the flat memory offsets for the 4x4 block of input values.
                # This matches: bs * (64 * 8 * 8) + ic * (8 * 8) + ih * 8 + iw
                input_offsets = (
                    bs * (IN_C * IN_H * IN_W)
                    + ic * (IN_H * IN_W)
                    + ih * IN_W
                    + iw
                )
                # Load the 4x4 block of input data.
                # Since the problem dimensions are fixed and all accesses are in-bounds,
                # no masking is required.
                input_vals = tl.load(input_ptr + input_offsets)

                # --- Kernel Index Calculation ---
                # The same kernel value is used for all 16 output pixels in this block.
                # This matches: oc * (64 * 2 * 2) + ic * (2 * 2) + kh * 2 + kw
                kernel_offset = (
                    oc * (IN_C * K_H * K_W)
                    + ic * (K_H * K_W)
                    + kh * K_W
                    + kw
                )
                # Load the single kernel value (scalar load).
                kernel_val = tl.load(kernel_ptr + kernel_offset)

                # --- Multiply and Accumulate ---
                # The scalar kernel_val is broadcasted to the 4x4 input_vals block.
                accumulator += input_vals * kernel_val

    # --- Output Index Calculation and Store ---
    # Calculate the flat memory offsets for the 4x4 output block.
    # This matches: bs * (128 * 4 * 4) + oc * (4 * 4) + oh * 4 + ow
    output_offsets = (
        bs * (OUT_C * OUT_H * OUT_W)
        + oc * (OUT_H * OUT_W)
        + oh * OUT_W
        + ow
    )
    # Store the final computed 4x4 block into the output tensor.
    tl.store(output_ptr + output_offsets, accumulator)


def triton_kernel(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    input_height: int,
    input_channels: int,
    output_channels: int,
    kernel_height: int,
    stride: int,
):
    """
    Wrapper function for the Triton convolution kernel.

    This function has a signature identical to the original CUDA wrapper,
    configures the launch grid, and calls the Triton JIT kernel.
    """
    # Calculate output dimensions, same as in the CUDA host code.
    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_height - kernel_height) // stride + 1  # Assuming square input

    # Define the grid for the kernel launch.
    # The CUDA launch used numBlocks(output_channels, 1, batch_size).
    # We map this to a 2D grid in Triton for clarity.
    grid = (output_channels, batch_size)

    # Launch the Triton kernel.
    # The tensors are passed directly; Triton handles extracting their data pointers.
    _triton_kernel_impl[grid](input, kernel, output)


if __name__ == "__main__":
    # --- Problem Definition ---
    # These parameters are derived from the CUDA code's loops and indexing.
    batch_size = 16
    input_channels = 64
    input_height = 8
    input_width = 8
    output_channels = 128
    kernel_height = 2
    kernel_width = 2
    stride = 2

    # Calculate output dimensions
    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1

    # --- Tensor Initialization ---
    # Set a seed for reproducibility
    torch.manual_seed(0)

    # Create input tensors on the GPU with NCHW layout
    input_tensor = torch.randn(
        (batch_size, input_channels, input_height, input_width),
        device="cuda",
        dtype=torch.float32,
    )

    # Create kernel tensor with (OC, IC, KH, KW) layout
    kernel_tensor = torch.randn(
        (output_channels, input_channels, kernel_height, kernel_width),
        device="cuda",
        dtype=torch.float32,
    )

    # Create an empty output tensor to store the Triton result
    triton_output = torch.empty(
        (batch_size, output_channels, output_height, output_width),
        device="cuda",
        dtype=torch.float32,
    )

    print("--- Input Shapes ---")
    print(f"Input:  {input_tensor.shape}")
    print(f"Kernel: {kernel_tensor.shape}")
    print(f"Output: {triton_output.shape}")
    print("-" * 20)

    # --- Kernel Execution ---
    print("Running Triton kernel...")
    triton_kernel(
        input_tensor,
        kernel_tensor,
        triton_output,
        batch_size,
        input_height,
        input_channels,
        output_channels,
        kernel_height,
        stride,
    )
    print("Triton kernel execution finished.")

    # --- Verification ---
    print("Running PyTorch reference kernel for verification...")
    # Use PyTorch's built-in conv2d as a ground truth
    torch_output = torch.nn.functional.conv2d(
        input_tensor, kernel_tensor, stride=stride
    )
    print("PyTorch reference execution finished.")

    # Compare the results
    # A small tolerance is used to account for potential floating-point arithmetic differences.
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-3):
        print("\n✅ SUCCESS: Triton kernel output matches PyTorch reference output.")
    else:
        print("\n❌ FAILED: Triton kernel output does NOT match PyTorch reference output.")
        # Print difference for debugging
        print("Max absolute difference:", torch.max(torch.abs(triton_output - torch_output)).item())