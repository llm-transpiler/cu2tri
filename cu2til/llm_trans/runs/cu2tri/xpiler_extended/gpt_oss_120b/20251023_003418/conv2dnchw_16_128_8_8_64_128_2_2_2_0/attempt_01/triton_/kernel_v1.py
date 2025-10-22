import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr, kernel_ptr, output_ptr,
    batch_size, input_height, input_channels,
    output_channels, kernel_height, stride,
    BLOCK_K: tl.constexpr
):
    # -------------------------------------------------
    # Derived dimensions (square spatial layout assumed)
    # -------------------------------------------------
    INPUT_W = input_height                # input width = input height
    KERNEL_W = kernel_height              # kernel width = kernel height
    output_height = (input_height - kernel_height) // stride + 1
    output_width = output_height          # square output

    # -------------------------------------------------
    # Program IDs
    # -------------------------------------------------
    ow = tl.program_id(0)                # output width index
    oh = tl.program_id(1)                # output height index
    pid_z = tl.program_id(2)             # combined batch and output channel

    # Decode batch (bs) and output channel (oc) from pid_z
    oc = pid_z % output_channels
    bs = pid_z // output_channels

    # Guard for out‑of‑range threads (should be unnecessary if grid matches dimensions)
    in_range = (bs < batch_size) & (oc < output_channels) & (oh < output_height) & (ow < output_width)

    # Linear offset of the output element
    out_offset = bs * (output_channels * output_height * output_width) + \
                 oc * (output_height * output_width) + \
                 oh * output_width + ow

    # Accumulator for the convolution sum
    acc = tl.zeros([1], dtype=tl.float32)

    # -------------------------------------------------
    # Spatial offsets for the 2×2 receptive field
    # -------------------------------------------------
    # Input offsets: top‑left corner + [0, 1, INPUT_W, INPUT_W+1]
    row_off_i = tl.arange(0, 2) * INPUT_W
    col_off_i = tl.arange(0, 2)
    input_spatial_offsets = (row_off_i[:, None] + col_off_i[None, :]).reshape(-1)   # shape (4,)

    # Kernel offsets: [0, 1, KERNEL_W, KERNEL_W+1] (KERNEL_W == 2)
    row_off_k = tl.arange(0, 2) * KERNEL_W
    col_off_k = tl.arange(0, 2)
    kernel_spatial_offsets = (row_off_k[:, None] + col_off_k[None, :]).reshape(-1)   # shape (4,)

    # -------------------------------------------------
    # Loop over input channels in tiles of size BLOCK_K
    # -------------------------------------------------
    ic = 0
    while ic < input_channels:
        cur_k = tl.minimum(input_channels - ic, BLOCK_K)   # actual tile size
        k = tl.arange(0, cur_k)                           # channel indices inside the tile

        # Base offset for the input tile:
        #   batch offset + channel offset + top‑left corner offset
        input_base = bs * (input_channels * INPUT_W * INPUT_W) + \
                     (ic + k) * (INPUT_W * INPUT_W) + \
                     (oh * stride) * INPUT_W + (ow * stride)

        # Base offset for the kernel tile:
        #   output‑channel offset + channel offset
        kernel_base = oc * (input_channels * KERNEL_W * KERNEL_W) + \
                      (ic + k) * (KERNEL_W * KERNEL_W)

        # Expand to (cur_k, 4) and add the spatial offsets
        input_offsets = input_base[:, None] + input_spatial_offsets[None, :]
        kernel_offsets = kernel_base[:, None] + kernel_spatial_offsets[None, :]

        # Load the 2×2 patches for all channels in the tile
        input_vals = tl.load(input_ptr + input_offsets, dtype=tl.float32)
        kernel_vals = tl.load(kernel_ptr + kernel_offsets, dtype=tl.float32)

        # Multiply‑accumulate over the tile
        acc += tl.sum(input_vals * kernel_vals)

        ic += BLOCK_K

    # -------------------------------------------------
    # Write the result
    # -------------------------------------------------
    tl.store(output_ptr + out_offset, acc, mask=in_range)


def triton_kernel(input_tensor, kernel_tensor, output_tensor,
                  batch_size, input_height,
                  input_channels, output_channels,
                  kernel_height, stride):
    """
    Triton implementation of the 2‑D convolution kernel.
    The signature matches the original CUDA kernel.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Shape (batch_size, input_channels, input_height, input_height)
    kernel_tensor : torch.Tensor
        Shape (output_channels, input_channels, kernel_height, kernel_height)
    output_tensor : torch.Tensor
        Shape (batch_size, output_channels, output_height, output_height)
    batch_size, input_height, input_channels, output_channels,
    kernel_height, stride : int
        Convolution parameters.
    """
    # -----------------------------------------------------------------
    # Basic sanity checks (CUDA tensors, contiguous layout)
    # -----------------------------------------------------------------
    assert input_tensor.is_cuda and input_tensor.is_contiguous()
    assert kernel_tensor.is_cuda and kernel_tensor.is_contiguous()
    assert output_tensor.is_cuda and output_tensor.is_contiguous()

    # -----------------------------------------------------------------
    # Compute output spatial dimensions
    # -----------------------------------------------------------------
    output_height = (input_height - kernel_height) // stride + 1
    output_width = output_height  # square output

    # Optional shape validation
    assert input_tensor.shape == (batch_size, input_channels, input_height, input_height)
    assert kernel_tensor.shape == (output_channels, input_channels, kernel_height, kernel_height)
    assert output_tensor.shape == (batch_size, output_channels, output_height, output_width)

    # -----------------------------------------------------------------
    # Grid configuration: (output_width, output_height, batch_size * output_channels)
    # -----------------------------------------------------------------
    grid = (output_width, output_height, batch_size * output_channels)

    # -----------------------------------------------------------------
    # Launch the Triton kernel
    # -----------------------------------------------------------------
    _triton_kernel_impl[grid](
        input_tensor,
        kernel_tensor,
        output_tensor,
        batch_size,
        input_height,
        input_channels,
        output_channels,
        kernel_height,
        stride,
        BLOCK_K=32  # tile size for the input‑channel dimension
    )
    # Ensure the kernel has finished before returning
    torch.cuda.synchronize()