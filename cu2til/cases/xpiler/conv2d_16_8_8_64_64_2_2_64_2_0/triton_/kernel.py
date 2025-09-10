import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    # Pointers to tensors
    input_ptr, kernel_ptr, output_ptr,
    # Stride parameters for input tensor (assumed NHWC layout)
    in_stride_n, in_stride_h, in_stride_w, in_stride_c,
    # Stride parameters for kernel tensor (assumed OKHWIc layout)
    k_stride_o, k_stride_h, k_stride_w, k_stride_i,
    # Stride parameters for output tensor (assumed NHWC layout)
    out_stride_n, out_stride_h, out_stride_w, out_stride_c,
    # Convolution parameters
    IC: tl.constexpr,
    OC: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    # Block size constants for tiling
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr
):
    """
    Triton kernel for 2D convolution based on the provided CUDA code.
    
    This kernel is designed with the following memory layouts in mind, derived from
    the index calculations in the original CUDA code:
    - Input: NHWC (Batch, Height, Width, Input Channels)
    - Kernel: OKHWIc (Output Channels, Kernel Height, Kernel Width, Input Channels)
    - Output: NHWC (Batch, Height, Width, Output Channels)

    Each program instance computes the output for one spatial location (n, oh, ow)
    across a block of output channels.
    """
    # Get program IDs which map to the output tensor's spatial dimensions
    pid_n = tl.program_id(axis=0)  # Batch size index
    pid_oh = tl.program_id(axis=1) # Output height index
    pid_ow = tl.program_id(axis=2) # Output width index

    # Offsets for the block of output channels this program will compute.
    # e.g., for BLOCK_OC=64, this is tl.arange(0, 64) -> [0, 1, ..., 63]
    offs_oc = tl.arange(0, BLOCK_OC)

    # Accumulator for the convolution sum, initialized to zeros.
    # We maintain one accumulator per output channel in the block.
    # It is shaped (BLOCK_OC,) for easier manual accumulation.
    accumulator = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over the kernel's spatial dimensions (height and width).
    # These loops are unrolled by the Triton compiler.
    for kh in range(KH):
        for kw in range(KW):
            # Calculate the corresponding input coordinates
            ih = pid_oh * SH + kh
            iw = pid_ow * SW + kw

            # Loop over the input channels in blocks of size BLOCK_IC
            for ic_start in range(0, IC, BLOCK_IC):
                offs_ic = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = offs_ic < IC

                # --- 1. Load Input Data ---
                # Load a vector of input values of size BLOCK_IC.
                # Pointer corresponds to input[pid_n, ih, iw, offs_ic].
                # Since input layout is NHWC, the channel dimension is contiguous.
                in_ptrs = (input_ptr +
                           pid_n * in_stride_n +
                           ih * in_stride_h +
                           iw * in_stride_w +
                           offs_ic * in_stride_c)
                # input_vals has shape (BLOCK_IC,)
                input_vals = tl.load(in_ptrs, mask=ic_mask, other=0.0)

                # --- 2. Load Kernel Data ---
                # Load a matrix of kernel weights of size (BLOCK_OC, BLOCK_IC).
                # Pointer corresponds to kernel[offs_oc, kh, kw, offs_ic].
                # Since kernel layout is OKHWIc, the input channel dim is contiguous.
                k_ptrs = (kernel_ptr +
                          offs_oc[:, None] * k_stride_o + # Broadcast for OC dimension
                          kh * k_stride_h +
                          kw * k_stride_w +
                          offs_ic[None, :] * k_stride_i)  # Broadcast for IC dimension
                
                k_mask = (offs_oc[:, None] < OC) & (ic_mask[None, :])
                # kernel_vals has shape (BLOCK_OC, BLOCK_IC)
                kernel_vals = tl.load(k_ptrs, mask=k_mask, other=0.0)

                # --- 3. Compute and Accumulate ---
                # The tl.dot operation has minimum size requirements (e.g., for tensor cores)
                # which are not met by a matrix-vector product (where one dimension is 1).
                # We perform the dot product manually using element-wise multiplication and summation.
                # Broadcast input_vals from (BLOCK_IC,) to (1, BLOCK_IC)
                # and multiply with kernel_vals of shape (BLOCK_OC, BLOCK_IC).
                # Then sum over the BLOCK_IC dimension.
                accumulator += tl.sum(kernel_vals * input_vals[None, :], axis=1)


    # --- 4. Store Output Data ---
    # Pointer corresponds to output[pid_n, pid_oh, pid_ow, offs_oc].
    out_ptrs = (output_ptr +
                pid_n * out_stride_n +
                pid_oh * out_stride_h +
                pid_ow * out_stride_w +
                offs_oc * out_stride_c)
    out_mask = offs_oc < OC
    tl.store(out_ptrs, accumulator, mask=out_mask)


def triton_kernel(input: torch.Tensor, filter: torch.Tensor, output: torch.Tensor,
                  batch_size: int, input_height: int,
                  input_channels: int, output_channels: int,
                  kernel_height: int, stride: int):
    """
    Wrapper function for the Triton convolution kernel, matching the CUDA host function signature.

    Args:
        input (torch.Tensor): Input tensor. Based on CUDA indexing, assumed to have
                              shape (N, H, W, C_in) and be contiguous in NHWC format.
        filter (torch.Tensor): Kernel tensor. Based on CUDA indexing, assumed to have
                               shape (C_out, KH, KW, C_in) and be contiguous in OKHWIc format.
        output (torch.Tensor): Output tensor, to be populated by the kernel.
                               Shape (N, OH, OW, C_out) in NHWC format.
        batch_size (int): Batch size (N).
        input_height (int): Height of the input tensor (H).
        input_channels (int): Number of input channels (C_in).
        output_channels (int): Number of output channels (C_out).
        kernel_height (int): Height of the kernel (KH).
        stride (int): Convolution stride, assumed to be the same for H and W.
    """
    # Assuming square inputs, kernels, and strides as implied by the CUDA function signature
    input_width = input_height
    kernel_width = kernel_height
    stride_h = stride_w = stride

    # Calculate output dimensions
    output_height = (input_height - kernel_height) // stride_h + 1
    output_width = (input_width - kernel_width) // stride_w + 1
    
    # Grid configuration: one program instance per output spatial location (n, oh, ow)
    grid = (batch_size, output_height, output_width)

    # Choose tiling block sizes.
    # Set BLOCK_OC to output_channels to compute all channels in a single program instance.
    BLOCK_OC = output_channels
    # Set BLOCK_IC to a power of 2 for performance. Since IC=64, 64 is a good choice.
    BLOCK_IC = 64
    
    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        # Tensors
        input, filter, output,
        # Input strides (NHWC)
        *input.stride(),
        # Kernel strides (OKHWIc)
        *filter.stride(),
        # Output strides (NHWC)
        *output.stride(),
        # Convolution parameters
        IC=input_channels,
        OC=output_channels,
        KH=kernel_height,
        KW=kernel_width,
        SH=stride_h,
        SW=stride_w,
        # Tiling parameters (constexpr)
        BLOCK_OC=BLOCK_OC,
        BLOCK_IC=BLOCK_IC,
        # Performance tuning
        num_warps=4,
        num_stages=2,
    )

# To make the code directly executable, we include a main block for testing and verification.
if __name__ == '__main__':
    # Problem dimensions from the CUDA code
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

    # Seed for reproducibility
    torch.manual_seed(0)

    # Check for CUDA device
    if not torch.cuda.is_available():
        print("CUDA device not found. This script requires a GPU.")
        exit()
    device = torch.device("cuda:0")

    # Create random input tensors on the GPU
    # The memory layouts are chosen to match the access patterns in the CUDA code
    # Input: NHWC layout
    input_nhwc = torch.randn(
        (BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS),
        dtype=torch.float32, device=device
    )
    
    # Filter/Kernel: OKHWIc layout (Output channels, Kernel Height, Kernel Width, Input channels)
    filter_okhic = torch.randn(
        (OUTPUT_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH, INPUT_CHANNELS),
        dtype=torch.float32, device=device
    )
    
    # Output tensor to be filled by the Triton kernel
    output_triton = torch.empty(
        (BATCH_SIZE, OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNELS),
        dtype=torch.float32, device=device
    )

    print("Running Triton kernel...")
    # Run the Triton kernel
    triton_kernel(
        input_nhwc, filter_okhic, output_triton,
        BATCH_SIZE, INPUT_HEIGHT, INPUT_CHANNELS,
        OUTPUT_CHANNELS, KERNEL_HEIGHT, STRIDE
    )
    
    # --- Verification using PyTorch's native conv2d ---
    print("Running PyTorch reference kernel for verification...")
    # PyTorch's conv2d expects NCHW input and OIHW filter layout.
    # We need to permute our tensors to match.
    input_nchw = input_nhwc.permute(0, 3, 1, 2)
    filter_oihw = filter_okhic.permute(0, 3, 1, 2)
    
    # Run PyTorch's convolution
    output_torch_nchw = torch.nn.functional.conv2d(
        input_nchw, filter_oihw, stride=STRIDE
    )
    
    # Permute the PyTorch output back to NHWC layout for comparison
    output_torch_nhwc = output_torch_nchw.permute(0, 2, 3, 1)

    # --- Compare the results ---
    print("Comparing Triton and PyTorch results...")
    # Use torch.allclose for robust floating-point comparison
    if torch.allclose(output_triton, output_torch_nhwc, atol=1e-4, rtol=1e-5):
        print("✅ Success: Triton kernel output matches PyTorch reference output.")
    else:
        print("❌ Error: Triton kernel output does NOT match PyTorch reference output.")
        # Print max difference for debugging
        max_diff = (output_triton - output_torch_nhwc).abs().max().item()
        print(f"   Max absolute difference: {max_diff}")