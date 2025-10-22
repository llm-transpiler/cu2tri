import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel implementing the reference CUDA convolution
# ---------------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    input_ptr,          # *float32
    kernel_ptr,         # *float32
    output_ptr,         # *float32
    B,                  # batch size (runtime)
    H_in,               # input height (runtime)
    W_in,               # input width  (runtime)
    C_in,               # input channels (runtime)
    C_out,              # output channels (runtime)
    stride,             # stride (runtime)
    H_out,              # output height (runtime)
    W_out,              # output width  (runtime)
    BLOCK_OC: tl.constexpr,   # compile‑time block size for output channels
    BLOCK_IC: tl.constexpr,   # compile‑time block size for input channels
    K_h: tl.constexpr,        # kernel height (compile‑time)
    K_w: tl.constexpr,        # kernel width  (compile‑time)
):
    # -----------------------------------------------------------------------
    # Program IDs: each program instance computes one output pixel (ow,oh) for
    # one batch element (bs) and processes BLOCK_OC output channels.
    # -----------------------------------------------------------------------
    ow = tl.program_id(0)  # output width index
    oh = tl.program_id(1)  # output height index
    bs = tl.program_id(2)  # batch index

    # Guard against out‑of‑bounds launches
    if ow >= W_out or oh >= H_out or bs >= B:
        return

    # -----------------------------------------------------------------------
    # Output‑channel handling
    # -----------------------------------------------------------------------
    oc = tl.arange(0, BLOCK_OC)               # [BLOCK_OC]
    oc_mask = oc < C_out                       # mask for valid output channels

    # Accumulator for the convolution result (float32)
    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # -----------------------------------------------------------------------
    # Offsets that are constant for the whole inner loop
    # -----------------------------------------------------------------------
    batch_offset = bs * H_in * W_in * C_in      # start of the batch in the input tensor

    # -----------------------------------------------------------------------
    # Convolution loops over kernel spatial dimensions
    # -----------------------------------------------------------------------
    for kh in range(K_h):                      # compile‑time unrolled
        ih = oh * stride + kh                   # input height index
        for kw in range(K_w):                  # compile‑time unrolled
            iw = ow * stride + kw               # input width index

            # Base offset of the current input pixel (before channel dimension)
            input_pixel_offset = batch_offset + ((ih * W_in + iw) * C_in)

            # Spatial offset inside the kernel for the current (kh,kw)
            kernel_spatial_offset = (kh * K_w + kw) * C_in

            # Base offset for the kernel slice for each output channel
            # Shape: [BLOCK_OC]
            kernel_base = oc * (K_h * K_w * C_in) + kernel_spatial_offset

            # -------------------------------------------------------------------
            # Loop over input channels in blocks of BLOCK_IC
            # -------------------------------------------------------------------
            ic = 0
            while ic < C_in:
                ic_range = tl.arange(0, BLOCK_IC) + ic          # [BLOCK_IC]
                ic_mask = ic_range < C_in                       # mask for valid input channels

                # -------------------- Load input values --------------------
                input_vals = tl.load(
                    input_ptr + input_pixel_offset + ic_range,
                    mask=ic_mask,
                    other=0.0,
                )                                                # [BLOCK_IC]

                # -------------------- Load kernel values -------------------
                # Kernel values shape: [BLOCK_OC, BLOCK_IC]
                kernel_mask = oc_mask[:, None] & ic_mask[None, :]
                kernel_vals = tl.load(
                    kernel_ptr + kernel_base[:, None] + ic_range[None, :],
                    mask=kernel_mask,
                    other=0.0,
                )                                                # [BLOCK_OC, BLOCK_IC]

                # -------------------- Accumulate ---------------------------
                # Multiply and sum over the input‑channel dimension
                acc += tl.sum(kernel_vals * input_vals[None, :], axis=1)  # [BLOCK_OC]

                ic += BLOCK_IC

    # -----------------------------------------------------------------------
    # Write the result to the output tensor
    # -----------------------------------------------------------------------
    out_offset = (
        bs * H_out * W_out * C_out
        + ((oh * W_out + ow) * C_out)
        + oc
    )
    tl.store(output_ptr + out_offset, acc, mask=oc_mask)


# ---------------------------------------------------------------------------
# Python wrapper that mimics the original CUDA kernel signature
# ---------------------------------------------------------------------------
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
    Triton implementation of the reference CUDA convolution kernel.
    The tensor layout is NHWC for both input and output, and
    the filter layout is (C_out, K_h, K_w, C_in).

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape (batch_size, input_height, input_height, input_channels)
    filter : torch.Tensor
        Convolution filter of shape (output_channels, kernel_height, kernel_height, input_channels)
    output : torch.Tensor
        Output tensor of shape (batch_size, output_height, output_width, output_channels)
    batch_size : int
        Number of batches (must match the first dimension of `input` and `output`)
    input_height : int
        Spatial height/width of the input (square)
    input_channels : int
        Number of input channels (C_in)
    output_channels : int
        Number of output channels (C_out)
    kernel_height : int
        Height (and width) of the convolution kernel (square)
    stride : int
        Stride of the convolution
    """
    # -----------------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------------
    assert input.is_cuda and filter.is_cuda and output.is_cuda, "All tensors must be on CUDA"
    assert input.dtype == torch.float32 and filter.dtype == torch.float32 and output.dtype == torch.float32, "All tensors must be float32"
    assert input.is_contiguous() and filter.is_contiguous() and output.is_contiguous(), "All tensors must be contiguous"

    # -----------------------------------------------------------------------
    # Derived output dimensions (square case)
    # -----------------------------------------------------------------------
    output_height = (input_height - kernel_height) // stride + 1
    output_width = output_height  # square output

    # -----------------------------------------------------------------------
    # Grid configuration: one program per (ow, oh, batch)
    # -----------------------------------------------------------------------
    grid = (output_width, output_height, batch_size)

    # -----------------------------------------------------------------------
    # Launch the Triton kernel
    # -----------------------------------------------------------------------
    _triton_kernel_impl[grid](
        input,
        filter,
        output,
        batch_size,
        input_height,          # H_in
        input_height,          # W_in (square)
        input_channels,
        output_channels,
        stride,
        output_height,
        output_width,
        BLOCK_OC=64,           # matches the CUDA blockDim.x
        BLOCK_IC=32,           # chosen to tile the 128 input channels efficiently
        K_h=kernel_height,
        K_w=kernel_height,
        num_warps=4,           # 4 warps per program (tuned for H800)
    )
    # Ensure kernel completion before returning to Python
    torch.cuda.synchronize()