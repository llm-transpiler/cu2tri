import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr, kernel_ptr, output_ptr,
    batch_size, input_channels, input_height,
    output_channels, output_height,
    kernel_height, stride,
    BLOCK_K: tl.constexpr,
    INPUT_CHANNELS: tl.constexpr):
    # Derived dimensions (square tensors)
    input_width = input_height
    output_width = output_height
    kernel_width = kernel_height

    # Program ID (1‑D grid)
    pid = tl.program_id(0)

    # Decode linear index into (bs, oc, oh, ow)
    ow = pid % output_width
    pid = pid // output_width

    oh = pid % output_height
    pid = pid // output_height

    oc = pid % output_channels
    pid = pid // output_channels

    bs = pid

    # Guard mask (true for valid threads)
    mask = (bs < batch_size) & (oc < output_channels) & (oh < output_height) & (ow < output_width)

    # Strides for contiguous NCHW layout
    input_stride_c = input_height * input_width          # H*W per input channel
    input_stride_n = input_channels * input_stride_c      # C*H*W per batch

    kernel_stride_c = kernel_height * kernel_width        # Kh*Kw per input channel

    output_stride_c = output_height * output_width        # H*W per output channel
    output_stride_n = output_channels * output_stride_c   # C*H*W per batch

    # Accumulator (scalar)
    acc = tl.zeros([1], dtype=tl.float32)[0]

    # Loop over input channels in blocks of BLOCK_K
    for ic in range(0, INPUT_CHANNELS, BLOCK_K):
        cur_ic = ic + tl.arange(0, BLOCK_K)
        ic_mask = cur_ic < INPUT_CHANNELS

        # Base offsets for this block of input channels
        input_base = bs * input_stride_n + cur_ic * input_stride_c
        kernel_base = oc * (INPUT_CHANNELS * kernel_stride_c) + cur_ic * kernel_stride_c

        # 2×2 kernel, stride as given
        for kh in range(2):
            for kw in range(2):
                ih = oh * stride + kh
                iw = ow * stride + kw

                # Offsets for the current spatial location
                input_offset = input_base + ih * input_width + iw
                kernel_offset = kernel_base + kh * kernel_width + kw

                # Load vectors (masked)
                load_mask = ic_mask & mask
                input_vals = tl.load(input_ptr + input_offset, mask=load_mask, other=0.0)
                kernel_vals = tl.load(kernel_ptr + kernel_offset, mask=load_mask, other=0.0)

                # Accumulate dot product over channel block
                acc += tl.dot(input_vals, kernel_vals)

    # Write the result
    output_offset = bs * output_stride_n + oc * output_stride_c + oh * output_width + ow
    tl.store(output_ptr + output_offset, acc, mask=mask)


def triton_kernel(input: torch.Tensor,
                  kernel: torch.Tensor,
                  output: torch.Tensor,
                  batch_size: int,
                  input_height: int,
                  input_channels: int,
                  output_channels: int,
                  kernel_height: int,
                  stride: int):
    """
    Triton wrapper reproducing the original CUDA kernel.
    Parameter order matches the original CUDA entry point.
    """
    # Compute output spatial dimensions (square)
    output_height = (input_height - kernel_height) // stride + 1
    output_width = output_height

    # Basic sanity checks
    assert input.is_cuda and kernel.is_cuda and output.is_cuda, "All tensors must be on CUDA"
    assert input.dtype == torch.float32 and kernel.dtype == torch.float32 and output.dtype == torch.float32, "All tensors must be float32"
    assert input.is_contiguous() and kernel.is_contiguous() and output.is_contiguous(), "All tensors must be contiguous"
    assert input.shape == (batch_size, input_channels, input_height, input_height), f"Unexpected input shape {input.shape}"
    assert kernel.shape == (output_channels, input_channels, kernel_height, kernel_height), f"Unexpected kernel shape {kernel.shape}"
    assert output.shape == (batch_size, output_channels, output_height, output_width), f"Unexpected output shape {output.shape}"

    # Tuning parameter: channel block size
    BLOCK_K = 32  # works well for 128 input channels on H800

    # Total number of output elements
    total_output = batch_size * output_channels * output_height * output_width

    # 1‑D grid: one program per output element
    grid = (total_output,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input, kernel, output,
        batch_size, input_channels, input_height,
        output_channels, output_height,
        kernel_height, stride,
        BLOCK_K=BLOCK_K,
        INPUT_CHANNELS=input_channels
    )