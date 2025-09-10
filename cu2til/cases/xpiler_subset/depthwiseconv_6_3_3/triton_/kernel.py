import torch
import triton
import triton.language as tl

# The actual Triton kernel (decorated with @triton.jit)
@triton.jit
def _triton_kernel_impl(
    # Pointers to tensors
    input_ptr,
    filter_ptr,
    output_ptr,
    # Stride information for each tensor (HWC layout)
    input_stride_h, input_stride_w, input_stride_c,
    filter_stride_kh, filter_stride_kw, filter_stride_c,
    output_stride_h, output_stride_w, output_stride_c,
    # Compile-time constants for dimensions for better optimization
    INPUT_H: tl.constexpr,
    INPUT_W: tl.constexpr,
    FILTER_H: tl.constexpr,
    FILTER_W: tl.constexpr,
    OUTPUT_H: tl.constexpr,
    OUTPUT_W: tl.constexpr,
    # Block sizes for tiling the output space
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """
    Triton kernel for 2D convolution on HWC data layout.
    This kernel mimics the logic of the provided CUDA code.
    Each program instance computes a BLOCK_SIZE_H x BLOCK_SIZE_W tile of the output
    for a single channel.
    """
    # --- Program IDs and Offsets ---
    # This program instance computes a tile of the output.
    # The grid is 3D, mapping to (output_H, output_W, channels).
    pid_h = tl.program_id(0)  # Grid dimension 0 -> tiles over output height
    pid_w = tl.program_id(1)  # Grid dimension 1 -> tiles over output width
    pid_c = tl.program_id(2)  # Grid dimension 2 -> maps to a specific channel

    # Calculate the offsets for the output block this program will handle.
    # tl.arange creates a 1D vector, e.g., [0, 1, 2, ..., BLOCK_SIZE_H-1].
    # Shape: (BLOCK_SIZE_H,)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    # Shape: (BLOCK_SIZE_W,)
    offs_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # --- Accumulator Initialization ---
    # Initialize a 2D block of accumulators to zero for the output tile.
    # Shape: (BLOCK_SIZE_H, BLOCK_SIZE_W)
    accumulator = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # --- Pointer Setup for the Current Channel ---
    # Advance base pointers to the correct channel based on program ID.
    input_ptr += pid_c * input_stride_c
    filter_ptr += pid_c * filter_stride_c

    # --- Convolution Loop ---
    # Iterate over the filter dimensions (3x3 in this case).
    # These standard Python loops are unrolled by the Triton compiler for performance.
    for kh in range(FILTER_H):
        for kw in range(FILTER_W):
            # --- Load Input Tile ---
            # Calculate the coordinates of the input tile corresponding to the filter position.
            input_offs_h = offs_h + kh
            input_offs_w = offs_w + kw

            # Create a 2D grid of pointers to the input tile.
            # Broadcasting `offs_h[:, None]` (column vector) and `offs_w[None, :]` (row vector)
            # creates a 2D matrix of offsets.
            input_ptrs = (input_ptr +
                          input_offs_h[:, None] * input_stride_h +
                          input_offs_w[None, :] * input_stride_w)

            # --- Load Filter Value ---
            # The filter value is a scalar for this iteration of the loop.
            filter_offset = kh * filter_stride_kh + kw * filter_stride_kw
            filter_val = tl.load(filter_ptr + filter_offset)

            # --- Boundary Checks (Masking) ---
            # Create a mask to prevent loading data from outside the input tensor's bounds.
            # This is crucial for correctness, especially at the edges.
            mask = (input_offs_h[:, None] < INPUT_H) & (input_offs_w[None, :] < INPUT_W)
            
            # Load the input tile using the mask. `other=0.0` ensures that any
            # out-of-bounds reads return 0.0, which is correct for convolution.
            input_vals = tl.load(input_ptrs, mask=mask, other=0.0)

            # --- Multiply-Accumulate ---
            # The core convolution operation.
            accumulator += input_vals * filter_val

    # --- Store Output Tile ---
    # Calculate pointers to the output tile for the current channel.
    output_ptr += pid_c * output_stride_c
    output_ptrs = (output_ptr +
                   offs_h[:, None] * output_stride_h +
                   offs_w[None, :] * output_stride_w)

    # Create a mask to prevent writing outside the output tensor's bounds.
    # This is necessary if the output dimensions are not perfectly divisible by block sizes.
    output_mask = (offs_h[:, None] < OUTPUT_H) & (offs_w[None, :] < OUTPUT_W)
    tl.store(output_ptrs, accumulator, mask=output_mask)


def triton_kernel(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor,
                  input_height: int, kernel_size: int, input_channels: int):
    """
    Wrapper function for the Triton 2D convolution kernel. This function has the
    same parameter signature as the original CUDA wrapper and handles grid
    configuration and kernel launch.

    Args:
        input (torch.Tensor): The input tensor with shape (H, W, C).
        kernel (torch.Tensor): The filter tensor with shape (KH, KW, C).
        output (torch.Tensor): The output tensor to store the result.
        input_height (int): The height of the input tensor (assumes square).
        kernel_size (int): The size of the filter (assumes square).
        input_channels (int): The number of channels.
    """
    # --- Shape and Dimension Calculations ---
    # These are derived from the CUDA code's logic.
    output_height = input_height - kernel_size + 1
    output_width = input_height - kernel_size + 1  # Assumes square input

    # --- Input Validation ---
    # Check that tensor shapes match the provided dimensions.
    assert input.shape == (input_height, input_height, input_channels)
    assert kernel.shape == (kernel_size, kernel_size, input_channels)
    assert output.shape == (output_height, output_width, input_channels)
    # Check for CUDA tensors and float32 dtype.
    assert all(t.is_cuda for t in [input, kernel, output])
    assert all(t.dtype == torch.float32 for t in [input, kernel, output])
    # Ensure tensors are contiguous in memory for correct stride calculations.
    # A contiguous HWC tensor of shape (H, W, C) has strides (W*C, C, 1).
    assert all(t.is_contiguous() for t in [input, kernel, output])

    # --- Grid and Block Configuration ---
    # Define a tile size for the output. 16x16 is a reasonable default,
    # similar to the original CUDA block size.
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    # The grid is 3D.
    # Dimensions 0 and 1 tile over the output height and width.
    # Dimension 2 corresponds to the channels, launching a separate program for each.
    grid = (
        triton.cdiv(output_width, BLOCK_SIZE_W),
        triton.cdiv(output_height, BLOCK_SIZE_H),
        input_channels
    )

    # --- Kernel Launch ---
    _triton_kernel_impl[grid](
        # Tensors
        input,
        kernel,
        output,
        # Strides
        input.stride(0), input.stride(1), input.stride(2),
        kernel.stride(0), kernel.stride(1), kernel.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        # Dimensions (passed as compile-time constants)
        INPUT_H=input_height,
        INPUT_W=input_height,
        FILTER_H=kernel_size,
        FILTER_W=kernel_size,
        OUTPUT_H=output_height,
        OUTPUT_W=output_width,
        # Block sizes (passed as compile-time constants)
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )


if __name__ == '__main__':
    # --- Problem Definition ---
    # Based on the CUDA code's hardcoded values
    input_height = 6
    input_width = 6
    kernel_size = 3
    input_channels = 3
    output_height = input_height - kernel_size + 1
    output_width = input_width - kernel_size + 1

    # --- Data Initialization ---
    # Use a fixed seed for reproducibility
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA device not found. Skipping execution.")
        exit()

    # Create tensors with HWC layout and ensure they are contiguous
    # Shape (H, W, C)
    input_tensor = torch.randn((input_height, input_width, input_channels), device=device, dtype=torch.float32).contiguous()
    # Shape (KH, KW, C)
    kernel_tensor = torch.randn((kernel_size, kernel_size, input_channels), device=device, dtype=torch.float32).contiguous()
    # Output tensor to be filled by the Triton kernel
    output_triton = torch.empty((output_height, output_width, input_channels), device=device, dtype=torch.float32).contiguous()

    # --- Execute Triton Kernel ---
    print("Executing Triton kernel...")
    triton_kernel(input_tensor, kernel_tensor, output_triton, input_height, kernel_size, input_channels)
    torch.cuda.synchronize()
    print("Triton kernel execution finished.")

    # --- Verification with PyTorch ---
    # PyTorch's conv2d expects NCHW layout, so we must permute our tensors.
    # The operation in the CUDA kernel is equivalent to a grouped convolution
    # where each input channel is convolved with its corresponding filter channel.
    print("Verifying with PyTorch's grouped convolution...")
    
    # Convert input from HWC to NCHW: (H, W, C) -> (C, H, W) -> (N, C, H, W)
    input_nchw = input_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # PyTorch conv2d filter format is (out_channels, in_channels/groups, KH, KW).
    # For our case, out_channels=C, in_channels=C, groups=C.
    # So we need to reshape our (KH, KW, C) kernel to (C, 1, KH, KW).
    # (KH, KW, C) -> (C, KH, KW) -> (C, 1, KH, KW)
    kernel_nchw = kernel_tensor.permute(2, 0, 1).unsqueeze(1)

    # Perform grouped convolution
    output_torch_nchw = torch.nn.functional.conv2d(input_nchw, kernel_nchw, groups=input_channels)
    
    # Convert PyTorch output back to HWC for comparison
    # (N, C, H, W) -> (C, H, W) -> (H, W, C)
    output_torch = output_torch_nchw.squeeze(0).permute(1, 2, 0)

    # --- Compare Results ---
    print("Comparing Triton and PyTorch results...")
    # Use torch.allclose for robust floating-point comparison
    are_close = torch.allclose(output_triton, output_torch, atol=1e-4, rtol=1e-4)
    print(f"Results are close: {are_close}")

    if not are_close:
        print("Mismatch detected!")
        print("Triton output norm:", torch.linalg.norm(output_triton))
        print("PyTorch output norm:", torch.linalg.norm(output_torch))
        print("Difference norm:", torch.linalg.norm(output_triton - output_torch))
        # For detailed debugging, uncomment the following lines:
        # print("Triton output:\n", output_triton)
        # print("PyTorch output:\n", output_torch)
        # print("Difference:\n", torch.abs(output_triton - output_torch))

    assert are_close, "Verification failed: Triton and PyTorch results do not match."
    print("\nVerification successful! The Triton kernel correctly implements the convolution.")