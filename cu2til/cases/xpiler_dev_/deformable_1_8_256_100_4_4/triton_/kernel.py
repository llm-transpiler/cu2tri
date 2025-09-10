import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel_impl(
    attention_weights_ptr,
    output_ptr,
    sampling_locations_ptr,
    value_ptr,
    value_level_start_index_ptr,
    value_spatial_shapes_ptr,
    # Tensor dimensions passed as compile-time constants for optimization
    N_QUERIES: tl.constexpr,
    N_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
    N_LEVELS: tl.constexpr,
    N_POINTS: tl.constexpr,
    # Block size for the feature dimension
    BLOCK_D: tl.constexpr,
):
    """
    Triton kernel for deformable attention sampling and interpolation.
    This kernel is equivalent to the provided CUDA implementation.
    """
    # Get program IDs for the current instance of the kernel.
    # This corresponds to the CUDA grid dimensions (lq, m) or (blockIdx.x, blockIdx.z).
    pid_lq = tl.program_id(0)  # Query dimension index
    pid_m = tl.program_id(1)   # Head dimension index

    # Create a block of offsets for the feature dimension (D_HEAD).
    # This corresponds to the parallel work done by threads in a CUDA block.
    d_offsets = tl.arange(0, BLOCK_D)

    # Initialize the accumulator for the output values.
    attention_sum = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # The outer loops from the CUDA kernel are unrolled here.
    # This is efficient in Triton as these are small, fixed-size loops.
    for i in range(N_LEVELS):
        # Load spatial shapes (height and width) for the current feature level.
        # These are scalar loads, shared across all threads in the D_HEAD dimension.
        h = tl.load(value_spatial_shapes_ptr + i * 2).to(tl.int32)
        w = tl.load(value_spatial_shapes_ptr + i * 2 + 1).to(tl.int32)

        # Load the start index for the current level in the flattened 'value' tensor.
        level_start_index = tl.load(value_level_start_index_ptr + i).to(tl.int32)

        for k in range(N_POINTS):
            # --- 1. Load sampling location and attention weight for the current point ---
            # Calculate offsets based on the multi-dimensional tensor layout.
            sl_base_offset = (pid_lq * (N_HEADS * N_LEVELS * N_POINTS * 2) +
                             pid_m * (N_LEVELS * N_POINTS * 2) +
                             i * (N_POINTS * 2) +
                             k * 2)
            aw_offset = (pid_lq * (N_HEADS * N_LEVELS * N_POINTS) +
                        pid_m * (N_LEVELS * N_POINTS) +
                        i * N_POINTS +
                        k)

            # Perform scalar loads for y, x coordinates and the attention weight.
            sampling_loc_y = tl.load(sampling_locations_ptr + sl_base_offset)
            sampling_loc_x = tl.load(sampling_locations_ptr + sl_base_offset + 1)
            attn_weight = tl.load(attention_weights_ptr + aw_offset)

            # --- 2. Calculate grid coordinates for bilinear interpolation ---
            # Convert normalized sampling locations to grid coordinates.
            x_grid = sampling_loc_x * h.to(tl.float32) - 0.5
            y_grid = sampling_loc_y * w.to(tl.float32) - 0.5

            # Find the integer coordinates of the 4 surrounding corner points.
            x0 = tl.floor(x_grid).to(tl.int32)
            x1 = x0 + 1
            y0 = tl.floor(y_grid).to(tl.int32)
            y1 = y0 + 1

            # --- 3. Load feature vectors from the 4 corner points with boundary checks ---
            # The 'value' tensor has a logical layout of (sum(H_l*W_l), N_HEADS, D_HEAD).
            # The stride for the spatial dimension is (N_HEADS * D_HEAD).
            SPATIAL_STRIDE = N_HEADS * D_HEAD

            # Define base pointer for the current head and feature block.
            head_feature_ptr = pid_m * D_HEAD + d_offsets

            # Calculate offsets and masks for each of the 4 corners.
            # Corner (x0, y0)
            offset00 = (level_start_index + x0 * w + y0) * SPATIAL_STRIDE + head_feature_ptr
            mask00 = (x0 >= 0) & (x0 < h) & (y0 >= 0) & (y0 < w)
            val00 = tl.load(value_ptr + offset00, mask=mask00, other=0.0)

            # Corner (x0, y1)
            offset01 = (level_start_index + x0 * w + y1) * SPATIAL_STRIDE + head_feature_ptr
            mask01 = (x0 >= 0) & (x0 < h) & (y1 >= 0) & (y1 < w)
            val01 = tl.load(value_ptr + offset01, mask=mask01, other=0.0)

            # Corner (x1, y0)
            offset10 = (level_start_index + x1 * w + y0) * SPATIAL_STRIDE + head_feature_ptr
            mask10 = (x1 >= 0) & (x1 < h) & (y0 >= 0) & (y0 < w)
            val10 = tl.load(value_ptr + offset10, mask=mask10, other=0.0)

            # Corner (x1, y1)
            offset11 = (level_start_index + x1 * w + y1) * SPATIAL_STRIDE + head_feature_ptr
            mask11 = (x1 >= 0) & (x1 < h) & (y1 >= 0) & (y1 < w)
            val11 = tl.load(value_ptr + offset11, mask=mask11, other=0.0)

            # --- 4. Perform bilinear interpolation ---
            # Calculate interpolation weights from fractional coordinates.
            w_x1 = x_grid - x0.to(tl.float32)
            w_x0 = 1.0 - w_x1
            w_y1 = y_grid - y0.to(tl.float32)
            w_y0 = 1.0 - w_y1

            # Compute the interpolated feature vector.
            interp_val = (val00 * w_x0 * w_y0 +
                          val10 * w_x1 * w_y0 +
                          val01 * w_x0 * w_y1 +
                          val11 * w_x1 * w_y1)

            # --- 5. Accumulate the weighted value ---
            attention_sum += interp_val * attn_weight

    # --- 6. Store the final accumulated result to the output tensor ---
    output_offset = pid_lq * (N_HEADS * D_HEAD) + pid_m * D_HEAD + d_offsets
    tl.store(output_ptr + output_offset, attention_sum)


def triton_kernel(value: torch.Tensor,
                  value_spatial_shapes: torch.Tensor,
                  level_start_index: torch.Tensor,
                  sampling_locations: torch.Tensor,
                  attention_weights: torch.Tensor,
                  output: torch.Tensor):
    """
    Wrapper function for the Triton kernel with a signature identical to the CUDA function.

    This function configures the grid and launches the `_triton_kernel_impl`.
    """
    # Extract dimensions from the original CUDA launch configuration comments
    # dim3 numBlocks(lq, n, m); -> lq=100, n=1, m=8
    # dim3 blockSize(d / 4); -> d=256, blockSize=64
    N_QUERIES = 100  # lq
    N_HEADS = 8      # m
    D_HEAD = 256     # d

    # Other dimensions are inferred from the CUDA kernel's loops
    N_LEVELS = 4     # from `for (int i = 0; i < 4; ++i)`
    N_POINTS = 4     # from `for (int k = 0; k < 4; ++k)`

    # Configure the launch grid. The CUDA grid was (lq, 1, m).
    # We map this to a 2D grid in Triton for simplicity.
    grid = (N_QUERIES, N_HEADS)

    # Basic validation of input tensors
    # assert all(t.is_cuda for t in [value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights, output])
    # assert value.dtype == torch.float32
    # assert value_spatial_shapes.dtype == torch.int32
    # assert level_start_index.dtype == torch.int32
    # assert sampling_locations.dtype == torch.float32
    # assert attention_weights.dtype == torch.float32
    # assert output.dtype == torch.float32

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        attention_weights,
        output,
        sampling_locations,
        value,
        level_start_index,
        value_spatial_shapes,
        # Pass dimensions as constexpr for better compiler optimization
        N_QUERIES=N_QUERIES,
        N_HEADS=N_HEADS,
        D_HEAD=D_HEAD,
        N_LEVELS=N_LEVELS,
        N_POINTS=N_POINTS,
        # The size of the block in the feature dimension
        BLOCK_D=D_HEAD,
    )


def test_kernel():
    """
    Provides a test case to verify the functionality of the Triton kernel.
    It creates random tensors with appropriate shapes and data types,
    runs the kernel, and performs a basic sanity check on the output.
    """
    # --- Test Configuration ---
    # Dimensions from the original CUDA code
    N_QUERIES = 100
    N_HEADS = 8
    D_HEAD = 256
    N_LEVELS = 4
    N_POINTS = 4

    # Define example spatial shapes for each feature level
    spatial_shapes_list = [(32, 48), (16, 24), (8, 12), (4, 6)]
    
    # --- Create Tensors on CUDA device ---
    device = 'cuda'
    
    # value_spatial_shapes: shape (N_LEVELS, 2), dtype int32
    value_spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.int32, device=device)
    
    # level_start_index: shape (N_LEVELS,), dtype int32
    level_start_index_list = [0]
    current_index = 0
    for h, w in spatial_shapes_list[:-1]:
        current_index += h * w
        level_start_index_list.append(current_index)
    level_start_index = torch.tensor(level_start_index_list, dtype=torch.int32, device=device)
    
    total_spatial_elements = sum(h * w for h, w in spatial_shapes_list)
    
    # value: shape (total_spatial_elements, N_HEADS, D_HEAD), dtype float32
    value_shape = (total_spatial_elements, N_HEADS, D_HEAD)
    value = torch.randn(value_shape, dtype=torch.float32, device=device)
    
    # sampling_locations: shape (N_QUERIES, N_HEADS, N_LEVELS, N_POINTS, 2), dtype float32
    # Values should be in [0, 1] range for normalized coordinates.
    sampling_locations_shape = (N_QUERIES, N_HEADS, N_LEVELS, N_POINTS, 2)
    sampling_locations = torch.rand(sampling_locations_shape, dtype=torch.float32, device=device)
    
    # attention_weights: shape (N_QUERIES, N_HEADS, N_LEVELS, N_POINTS), dtype float32
    attention_weights_shape = (N_QUERIES, N_HEADS, N_LEVELS, N_POINTS)
    attention_weights = torch.randn(attention_weights_shape, dtype=torch.float32, device=device)
    
    # output: shape (N_QUERIES, N_HEADS, D_HEAD), dtype float32
    output_shape = (N_QUERIES, N_HEADS, D_HEAD)
    output = torch.empty(output_shape, dtype=torch.float32, device=device)

    print("--- Input Tensor Shapes ---")
    print(f"value: {value.shape}")
    print(f"value_spatial_shapes: {value_spatial_shapes.shape}")
    print(f"level_start_index: {level_start_index.shape}")
    print(f"sampling_locations: {sampling_locations.shape}")
    print(f"attention_weights: {attention_weights.shape}")
    print(f"output: {output.shape}")
    print("-" * 25)

    # --- Run Triton Kernel ---
    # Triton handles contiguous PyTorch tensors directly.
    triton_kernel(
        value,
        value_spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        output
    )
    
    # --- Print and Verify Results ---
    print("Triton kernel executed successfully.")
    print("Sample of output tensor (output[0, 0, :8]):")
    print(output[0, 0, :8])
    
    # A simple check to ensure the output tensor was written to.
    assert not torch.all(output == 0)
    print("\nVerification: Output tensor is not all zeros. OK.")


if __name__ == "__main__":
    # Set a seed for reproducibility of random inputs
    torch.manual_seed(42)
    test_kernel()