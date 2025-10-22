import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr,
    kernel_ptr,
    output_ptr,
    B,
    H_in,
    W_in,
    C_in,
    C_out,
    K_h,
    K_w,
    stride,
    H_out,
    W_out,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # Program IDs
    ow = tl.program_id(0)  # output width index
    oh = tl.program_id(1)  # output height index
    bs = tl.program_id(2)  # batch index

    # Bounds check
    if ow >= W_out or oh >= H_out or bs >= B:
        return

    # Output channel range for this program
    oc = tl.arange(0, BLOCK_OC)
    oc_mask = oc < C_out

    # Accumulator for the convolution result
    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # Offset to the beginning of the batch in the input tensor
    batch_offset = bs * H_in * W_in * C_in

    # Loop over kernel spatial dimensions
    for kh in range(K_h):
        ih = oh * stride + kh
        for kw in range(K_w):
            iw = ow * stride + kw

            # Base offset for the input pixel (before channel dimension)
            input_pixel_offset = batch_offset + ((ih * W_in + iw) * C_in)

            # Base offset for the kernel slice for each output channel
            kernel_spatial_offset = (kh * K_w + kw) * C_in
            kernel_base = oc * (K_h * K_w * C_in) + kernel_spatial_offset

            # Iterate over input channels in blocks
            for ic in range(0, C_in, BLOCK_IC):
                ic_range = tl.arange(0, BLOCK_IC) + ic
                ic_mask = ic_range < C_in

                # Load input values (shape [BLOCK_IC])
                input_vals = tl.load(
                    input_ptr + input_pixel_offset + ic_range,
                    mask=ic_mask,
                    other=0.0,
                )

                # Load kernel values (shape [BLOCK_OC, BLOCK_IC])
                kernel_offset = kernel_base[:, None] + ic_range[None, :]
                kernel_vals = tl.load(
                    kernel_ptr + kernel_offset,
                    mask=ic_mask[None, :],
                    other=0.0,
                )

                # Accumulate dot product over the current channel block
                acc += tl.dot(kernel_vals, input_vals)

    # Write the result to the output tensor
    out_offset = (
        bs * H_out * W_out * C_out
        + ((oh * W_out + ow) * C_out)
        + oc
    )
    tl.store(output_ptr + out_offset, acc, mask=oc_mask)


def triton_kernel(
    input: torch.Tensor,
    filter: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    input_height: int,
    input_channels: int,
    output_channels: int,
    kernel_height: int,
    stride: int,
):
    """
    Triton implementation of the reference CUDA kernel.
    All tensors must be contiguous float32 on the same CUDA device.
    """
    # Basic sanity checks
    assert input.is_cuda and filter.is_cuda and output.is_cuda, "All tensors must be on CUDA"
    assert input.dtype == torch.float32 and filter.dtype == torch.float32 and output.dtype == torch.float32, "All tensors must be float32"
    assert input.is_contiguous() and filter.is_contiguous() and output.is_contiguous(), "All tensors must be contiguous"

    # Derived spatial dimensions (assuming square input and kernel)
    output_height = (input_height - kernel_height) // stride + 1
    output_width = output_height  # square output

    # Grid configuration: (output_width, output_height, batch)
    grid = (output_width, output_height, batch_size)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        filter,
        output,
        batch_size,
        input_height,          # H_in
        input_height,          # W_in (square)
        input_channels,
        output_channels,
        kernel_height,
        kernel_height,         # K_w (square)
        stride,
        output_height,
        output_width,
        BLOCK_OC=64,
        BLOCK_IC=32,
        num_warps=4,
    )
    # Optional synchronization for correctness in a pure-Python context
    torch.cuda.synchronize()