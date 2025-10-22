import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing the same 3×3 per‑channel convolution as the CUDA kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    input_ptr, filter_ptr, output_ptr,
    input_height, input_channels,
    output_height, output_width,
    kernel_size: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr):
    """
    Arguments
    ----------
    input_ptr      : pointer to the input tensor (float32)
    filter_ptr     : pointer to the filter tensor (float32)
    output_ptr     : pointer to the output tensor (float32)
    input_height   : height (and width) of the input feature map
    input_channels : number of channels (C)
    output_height  : height of the output feature map
    output_width   : width of the output feature map
    kernel_size    : spatial size of the filter (compile‑time constant, 3 in this case)
    BLOCK_SIZE_X   : number of threads along X within a block (compile‑time constant)
    BLOCK_SIZE_Y   : number of threads along Y within a block (compile‑time constant)
    """
    pid_x = tl.program_id(0)  # block index in X dimension
    pid_y = tl.program_id(1)  # block index in Y dimension
    pid_c = tl.program_id(2)  # block index for channel (C)

    # ------------------------------------------------------------------
    # Compute the absolute coordinates of the threads within the output map
    # ------------------------------------------------------------------
    x = tl.arange(0, BLOCK_SIZE_X) + pid_x * BLOCK_SIZE_X   # shape (BLOCK_SIZE_X,)
    y = tl.arange(0, BLOCK_SIZE_Y) + pid_y * BLOCK_SIZE_Y   # shape (BLOCK_SIZE_Y,)

    # Broadcast to a 2‑D grid of coordinates
    X = x[None, :]   # (1, BLOCK_SIZE_X)
    Y = y[:, None]   # (BLOCK_SIZE_Y, 1)

    # Mask for threads that fall inside the valid output region
    mask = (X < output_width) & (Y < output_height)

    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)

    # ---------------------------------------------------------------
    # Perform the 3×3 convolution (kernel_size is a compile‑time constant)
    # ---------------------------------------------------------------
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Coordinates of the corresponding input element
            in_x = X + j
            in_y = Y + i

            # Flat offset into the input tensor:
            # ((in_y * input_height) + in_x) * input_channels + pid_c
            input_offset = (in_y * input_height + in_x) * input_channels + pid_c
            inp = tl.load(input_ptr + input_offset, mask=mask, other=0.0)

            # Flat offset into the filter tensor (scalar per channel)
            filter_offset = ((i * kernel_size + j) * input_channels) + pid_c
            filt = tl.load(filter_ptr + filter_offset)

            # Accumulate the product
            acc += inp * filt

    # Compute the flat offset for the output tensor
    out_offset = (Y * output_width + X) * input_channels + pid_c
    tl.store(output_ptr + out_offset, acc, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper that mirrors the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(input: torch.Tensor,
                  kernel: torch.Tensor,
                  output: torch.Tensor,
                  input_height: int,
                  kernel_size: int,
                  input_channels: int):
    """
    Entry‑point that launches the Triton kernel.
    Parameters match the original CUDA kernel:
        input          – (input_height, input_height, input_channels) float32 tensor
        kernel         – (kernel_size, kernel_size, input_channels) float32 tensor
        output         – (output_height, output_width, input_channels) float32 tensor
        input_height   – height (and width) of the input feature map
        kernel_size    – spatial size of the filter (must be 3 for this implementation)
        input_channels – number of channels (C)
    """
    # ------------------------------------------------------------------
    # Basic sanity checks
    # ------------------------------------------------------------------
    assert input.is_cuda and kernel.is_cuda and output.is_cuda, "All tensors must reside on CUDA"
    assert input.dtype == torch.float32 and kernel.dtype == torch.float32 and output.dtype == torch.float32, \
        "Only float32 tensors are supported"
    assert input.is_contiguous() and kernel.is_contiguous() and output.is_contiguous(), \
        "Tensors must be contiguous"

    # ------------------------------------------------------------------
    # Derive output dimensions (square input → square output)
    # ------------------------------------------------------------------
    output_height = input_height - kernel_size + 1
    output_width = output_height

    # ------------------------------------------------------------------
    # Validate tensor shapes
    # ------------------------------------------------------------------
    assert input.shape == (input_height, input_height, input_channels), \
        f"Expected input shape ({input_height},{input_height},{input_channels}), got {input.shape}"
    assert kernel.shape == (kernel_size, kernel_size, input_channels), \
        f"Expected kernel shape ({kernel_size},{kernel_size},{input_channels}), got {kernel.shape}"
    assert output.shape == (output_height, output_width, input_channels), \
        f"Expected output shape ({output_height},{output_width},{input_channels}), got {output.shape}"

    # ------------------------------------------------------------------
    # Kernel launch configuration
    # ------------------------------------------------------------------
    BLOCK_SIZE_X = 32
    BLOCK_SIZE_Y = 32

    grid = (
        (output_width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X,
        (output_height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y,
        input_channels,
    )

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        input,
        kernel,
        output,
        input_height,
        input_channels,
        output_height,
        output_width,
        kernel_size,          # constexpr kernel size (3)
        BLOCK_SIZE_X,
        BLOCK_SIZE_Y,
        num_warps=8,          # tuned for the H800 architecture
    )
    # Synchronize for correctness when the function returns
    torch.cuda.synchronize()