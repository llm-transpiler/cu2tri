import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    # Pointers to tensors
    input_ptr, kernel_ptr, output_ptr,
    # Input tensor strides
    IN_S_BS, IN_S_H, IN_S_W, IN_S_C,
    # Kernel tensor strides
    KERNEL_S_OC, KERNEL_S_H, KERNEL_S_W, KERNEL_S_C,
    # Output tensor strides
    OUT_S_BS, OUT_S_H, OUT_S_W, OUT_S_C,
    # Convolution parameters
    IC: tl.constexpr,
    OC: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    # Tiling parameters
    BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for 2D convolution, translated from the provided CUDA code.
    
    This kernel computes a tile of the output tensor. Each program instance is responsible
    for one output pixel location (ow, oh) within a specific batch (bs), and it calculates
    the values for all output channels (OC) for that pixel.

    The memory layout assumptions are:
    - Input: NHWC (Batch, Height, Width, Channel)
    - Kernel: OHWI (Output Channel, Height, Width, Input Channel)
    - Output: NHWC (Batch, Height, Width, Channel)
    """
    # --- Grid and Program IDs ---
    # The grid is 3D, mapping to (output_width, output_height, batch_size).
    pid_ow = tl.program_id(0)  # Output width index, from blockIdx.x
    pid_oh = tl.program_id(1)  # Output height index, from blockIdx.y
    pid_bs = tl.program_id(2)  # Batch index, from blockIdx.z

    # --- Accumulator ---
    # Initialize accumulators for all output channels for the current output pixel.
    # This corresponds to the 'sum' variable in the CUDA kernel, but vectorized
    # for all output channels (which were handled by threadIdx.x in CUDA).
    # Shape: (OC,)
    acc = tl.zeros((OC,), dtype=tl.float32)

    # --- Pointers and Offsets ---
    # Create a range of offsets for the output channels.
    offs_oc = tl.arange(0, OC)
    
    # The reduction is performed over the combined (KH, KW, IC) dimensions.
    K_DIM = KH * KW * IC

    # --- Main Reduction Loop ---
    # This loop replaces the three nested for-loops (kh, kw, ic) in the CUDA code.
    # It iterates over the combined reduction dimension K_DIM in blocks of BLOCK_K.
    for k_start in range(0, K_DIM, BLOCK_K):
        # Create a block of offsets for the current reduction step.
        offs_k = tl.arange(0, BLOCK_K)
        k = k_start + offs_k
        
        # Create a mask to handle cases where K_DIM is not a multiple of BLOCK_K.
        # This ensures we don't read or compute with out-of-bounds values.
        k_mask = (k < K_DIM)

        # Deconstruct the 1D reduction index 'k' into 3D (kh, kw, ic) indices.
        ic = k % IC
        rem = k // IC
        kw = rem % KW
        kh = rem // KW
        
        # --- Load Input Data ---
        # Calculate the corresponding input coordinates (ih, iw).
        ih = pid_oh * STRIDE_H + kh
        iw = pid_ow * STRIDE_W + kw
        
        # Calculate the memory offsets for the input block.
        # This is a 1D block of size BLOCK_K.
        input_offsets = (pid_bs * IN_S_BS + 
                         ih * IN_S_H + 
                         iw * IN_S_W + 
                         ic * IN_S_C)
        
        # Load a block of input values. Apply the mask to avoid out-of-bounds access.
        # The value loaded here is used for all output channels in this step.
        # Shape: (BLOCK_K,)
        input_block = tl.load(input_ptr + input_offsets, mask=k_mask, other=0.0)
        
        # --- Load Kernel Data ---
        # Calculate the memory offsets for the kernel block.
        # This is a 2D block of shape (BLOCK_K, OC).
        kernel_offsets = (offs_oc[None, :] * KERNEL_S_OC + 
                          kh[:, None] * KERNEL_S_H + 
                          kw[:, None] * KERNEL_S_W + 
                          ic[:, None] * KERNEL_S_C)
                          
        # Load a block of kernel values. Apply the mask to avoid out-of-bounds access.
        # Shape: (BLOCK_K, OC)
        kernel_block = tl.load(kernel_ptr + kernel_offsets, mask=k_mask[:, None], other=0.0)
        
        # --- Compute ---
        # Perform the element-wise multiplication and sum for this block.
        # 1. Broadcast `input_block` from (BLOCK_K,) to (BLOCK_K, OC).
        # 2. Multiply with `kernel_block` (BLOCK_K, OC).
        # 3. Sum the results over the reduction dimension (axis=0) to get a (OC,) vector.
        # 4. Add the result to the accumulator.
        acc += tl.sum(input_block[:, None] * kernel_block, axis=0)

    # --- Store Output ---
    # Calculate the memory offsets for the output pixel.
    output_offsets = (pid_bs * OUT_S_BS + 
                      pid_oh * OUT_S_H + 
                      pid_ow * OUT_S_W + 
                      offs_oc * OUT_S_C)
                      
    # Store the final accumulated values to the output tensor.
    tl.store(output_ptr + output_offsets, acc)


def triton_kernel(input: torch.Tensor, filter: torch.Tensor, output: torch.Tensor, 
                  batch_size: int, input_height: int,
                  input_channels: int, output_channels: int,
                  kernel_height: int, stride: int):
    """
    Wrapper function for the Triton 2D convolution kernel.

    This function sets up the grid and launches the Triton kernel, providing it
    with the necessary tensor pointers, strides, and convolution parameters.
    It has a signature identical to the original CUDA wrapper function.
    """
    # Basic validation
    assert input.is_cuda and filter.is_cuda and output.is_cuda, "All tensors must be on a CUDA device."
    assert input.dtype == torch.float32 and filter.dtype == torch.float32 and output.dtype == torch.float32, "All tensors must be of type float32."

    # Calculate output dimensions based on input and kernel parameters
    output_height = (input_height - kernel_height) // stride + 1
    # The original CUDA code uses input_height for both width and height calculations
    output_width = (input_height - kernel_height) // stride + 1

    # Define the grid for the kernel launch.
    # This matches the `dim3 numBlocks(output_width, output_height, batch_size)` in CUDA.
    grid = (output_width, output_height, batch_size)

    # Get tensor strides. The CUDA kernel implies specific memory layouts:
    # Input: NHWC (bs, ih, iw, ic)
    # Kernel: OHWI (oc, kh, kw, ic)
    # Output: NHWC (bs, oh, ow, oc)
    # We assume the input tensors are already in the correct contiguous format.
    
    # Input strides
    IN_S_BS, IN_S_H, IN_S_W, IN_S_C = input.stride()
    # Kernel strides
    KERNEL_S_OC, KERNEL_S_H, KERNEL_S_W, KERNEL_S_C = filter.stride()
    # Output strides
    OUT_S_BS, OUT_S_H, OUT_S_W, OUT_S_C = output.stride()

    # Choose a tiling size for the reduction dimension. This is a tunable parameter.
    # 64 is a reasonable starting point for balancing parallelism and register usage.
    BLOCK_K = 64
    
    # Launch the Triton kernel.
    _triton_kernel_impl[grid](
        # Pointers
        input, filter, output,
        # Input strides
        IN_S_BS, IN_S_H, IN_S_W, IN_S_C,
        # Kernel strides
        KERNEL_S_OC, KERNEL_S_H, KERNEL_S_W, KERNEL_S_C,
        # Output strides
        OUT_S_BS, OUT_S_H, OUT_S_W, OUT_S_C,
        # Convolution parameters (passed as tl.constexpr for compile-time optimization)
        IC=input_channels,
        OC=output_channels,
        KH=kernel_height,
        KW=kernel_height, # Assuming square kernel from CUDA code
        STRIDE_H=stride,
        STRIDE_W=stride, # Assuming square stride from CUDA code
        # Tiling parameters (as tl.constexpr)
        BLOCK_K=BLOCK_K,
    )


# To make the code directly executable, we include a test and verification block.
if __name__ == '__main__':
    import torch.nn.functional as F

    # --- Configuration based on the CUDA code's hardcoded values ---
    BATCH_SIZE = 16
    INPUT_HEIGHT = 8
    INPUT_WIDTH = 8
    INPUT_CHANNELS = 64
    OUTPUT_CHANNELS = 64
    KERNEL_HEIGHT = 2
    KERNEL_WIDTH = 2
    STRIDE = 2
    
    # Calculate output dimensions
    OUTPUT_HEIGHT = (INPUT_HEIGHT - KERNEL_HEIGHT) // STRIDE + 1
    OUTPUT_WIDTH = (INPUT_WIDTH - KERNEL_WIDTH) // STRIDE + 1

    # --- Data Initialization ---
    # Set a seed for reproducibility
    torch.manual_seed(0)
    device = 'cuda'

    # Create input tensor in NCHW format first, as it's standard for PyTorch
    input_nchw = torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, dtype=torch.float32, device=device)
    # Convert to NHWC (channels-last) format, which the kernel expects
    input_nhwc = input_nchw.contiguous(memory_format=torch.channels_last)

    # Create kernel tensor in OIHW format first (PyTorch standard)
    kernel_oihw = torch.randn(OUTPUT_CHANNELS, INPUT_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH, dtype=torch.float32, device=device)
    # Convert to OHWI format, which the kernel expects
    # (oc, ic, kh, kw) -> (oc, kh, kw, ic)
    kernel_ohwi = kernel_oihw.permute(0, 2, 3, 1).contiguous()

    # Create an empty output tensor for the Triton kernel to write to
    # It must be in NHWC format to match the kernel's output layout
    output_triton = torch.empty(BATCH_SIZE, OUTPUT_CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH, dtype=torch.float32, device=device)
    output_triton = output_triton.contiguous(memory_format=torch.channels_last)

    # --- Execute Triton Kernel ---
    print("Running Triton kernel...")
    triton_kernel(
        input=input_nhwc, 
        filter=kernel_ohwi, 
        output=output_triton,
        batch_size=BATCH_SIZE,
        input_height=INPUT_HEIGHT,
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        kernel_height=KERNEL_HEIGHT,
        stride=STRIDE
    )
    print("Triton kernel execution finished.")

    # --- Execute PyTorch Reference Implementation ---
    print("Running PyTorch reference kernel...")
    # F.conv2d expects NCHW input and OIHW kernel
    output_ref_nchw = F.conv2d(input_nchw, kernel_oihw, stride=STRIDE)
    # Convert reference output to NHWC for comparison
    output_ref_nhwc = output_ref_nchw.contiguous(memory_format=torch.channels_last)
    print("PyTorch reference execution finished.")

    # --- Verification ---
    print("Verifying results...")
    # Compare the Triton output with the PyTorch reference
    # We use a small tolerance (atol) for floating-point comparisons
    is_correct = torch.allclose(output_triton, output_ref_nhwc, atol=1e-3, rtol=1e-4)
    
    if is_correct:
        print("✅ Triton kernel output matches the PyTorch reference.")
    else:
        print("❌ Triton kernel output DOES NOT match the PyTorch reference.")
        # Print detailed error for debugging
        diff = torch.abs(output_triton - output_ref_nhwc)
        print(f"   Max absolute error: {diff.max().item()}")
        print(f"   Max relative error: {(diff / torch.abs(output_ref_nhwc)).max().item()}")