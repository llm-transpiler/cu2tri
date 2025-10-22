import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing the exact computation of the original CUDA
# kernel.  The layout is NHWC for the input and output tensors and
# (OC, Kh, Kw, C) for the filter tensor.
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    input_ptr,          # float*  (NHWC)
    kernel_ptr,         # float*  (OC, Kh, Kw, C)
    output_ptr,         # float*  (NHWC)
    batch_size,         # int
    input_h,            # int
    input_w,            # int
    stride,             # int
    output_h,           # int
    output_w,           # int
    BLOCK_SIZE: tl.constexpr,   # threads per program (= output channels)
    input_c: tl.constexpr,      # input channels (C)
    output_c: tl.constexpr,     # output channels (OC)
    kernel_h: tl.constexpr,     # kernel spatial size (Kh = Kw)
):
    # ------------------------------------------------------------------
    # Grid indices (NHWC layout)
    #   ow – output width  (program_id 0)
    #   oh – output height (program_id 1)
    #   bs – batch index   (program_id 2)
    # ------------------------------------------------------------------
    ow = tl.program_id(0)
    oh = tl.program_id(1)
    bs = tl.program_id(2)

    # ------------------------------------------------------------------
    # Thread index for the output‑channel dimension
    # ------------------------------------------------------------------
    oc = tl.arange(0, BLOCK_SIZE)          # [BLOCK_SIZE]
    oc_mask = oc < output_c                  # [BLOCK_SIZE] bool

    # ------------------------------------------------------------------
    # Accumulator for each output channel handled by this program
    # ------------------------------------------------------------------
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # ------------------------------------------------------------------
    # Input‑channel indices (vectorized)
    # ------------------------------------------------------------------
    ic = tl.arange(0, input_c)              # [input_c]

    # ------------------------------------------------------------------
    # Strides for the input tensor (NHWC)
    # ------------------------------------------------------------------
    input_batch_stride = input_h * input_w * input_c
    input_row_stride   = input_w * input_c
    input_col_stride   = input_c

    # ------------------------------------------------------------------
    # Strides for the kernel tensor (OC, Kh, Kw, C)
    # ------------------------------------------------------------------
    kernel_oc_stride = kernel_h * kernel_h * input_c   # OC stride
    kernel_kh_stride = kernel_h * input_c              # Kh stride
    kernel_kw_stride = input_c                         # Kw stride

    # ------------------------------------------------------------------
    # Convolution loops (Kh = Kw = kernel_h, here 2)
    # ------------------------------------------------------------------
    for kh in range(kernel_h):
        for kw in range(kernel_h):
            # Spatial location in the input that corresponds to (oh, ow, kh, kw)
            ih = oh * stride + kh
            iw = ow * stride + kw

            # ------------------------------------------------------------------
            # Load the input vector of length C for this (ih, iw)
            # ------------------------------------------------------------------
            input_base = (bs * input_batch_stride
                          + ih * input_row_stride
                          + iw * input_col_stride)
            # No mask needed – the indices are guaranteed to be in‑bounds
            input_vec = tl.load(input_ptr + input_base + ic)   # [C]

            # Broadcast to shape [1, C] for element‑wise multiply with the kernel
            input_mat = input_vec[None, :]                     # [1, C]

            # ------------------------------------------------------------------
            # Load the kernel matrix for all OC handled by this program
            # ------------------------------------------------------------------
            kernel_base = (oc[:, None] * kernel_oc_stride
                           + kh * kernel_kh_stride
                           + kw * kernel_kw_stride)        # [BLOCK_SIZE, 1]
            kernel_mat = tl.load(kernel_ptr + kernel_base + ic[None, :],
                                 mask=oc_mask[:, None],
                                 other=0.0)                    # [BLOCK_SIZE, C]

            # ------------------------------------------------------------------
            # Accumulate dot product over the channel dimension
            # ------------------------------------------------------------------
            acc += tl.sum(input_mat * kernel_mat, axis=1)      # [BLOCK_SIZE]

    # ------------------------------------------------------------------
    # Strides for the output tensor (NHWC)
    # ------------------------------------------------------------------
    output_batch_stride = output_h * output_w * output_c
    output_row_stride   = output_w * output_c
    output_col_stride   = output_c

    # ------------------------------------------------------------------
    # Store the result
    # ------------------------------------------------------------------
    output_base = (bs * output_batch_stride
                   + oh * output_row_stride
                   + ow * output_col_stride)
    tl.store(output_ptr + output_base + oc, acc, mask=oc_mask)


# ----------------------------------------------------------------------
# Python wrapper matching the original CUDA kernel signature.
# ----------------------------------------------------------------------
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
    Triton implementation equivalent to the original CUDA kernel.
    Expected tensor layouts (NHWC):
        input  : [batch, H, W, C]
        filter : [OC, Kh, Kw, C]
        output : [batch, OH, OW, OC]
    """
    # ------------------------------------------------------------------
    # Basic sanity checks
    # ------------------------------------------------------------------
    if not (input.is_cuda and filter.is_cuda and output.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    input = input.contiguous()
    filter = filter.contiguous()
    output = output.contiguous()

    # ------------------------------------------------------------------
    # Compute output spatial dimensions (square case, as in the CUDA code)
    # ------------------------------------------------------------------
    output_height = (input_height - kernel_height) // stride + 1
    output_width = output_height  # assumes square input / output

    # ------------------------------------------------------------------
    # Grid configuration: (output_width, output_height, batch)
    # ------------------------------------------------------------------
    grid = (output_width, output_height, batch_size)

    # ------------------------------------------------------------------
    # Launch the Triton kernel.
    # Compile‑time constants are passed as keyword arguments.
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        input,
        filter,
        output,
        batch_size,
        input_height,          # input_h
        input_height,          # input_w (square)
        stride,
        output_height,
        output_width,
        BLOCK_SIZE=output_channels,   # threads per program (must be multiple of 32)
        input_c=input_channels,
        output_c=output_channels,
        kernel_h=kernel_height,
        num_warps=2,                  # 2 warps = 64 threads (matches BLOCK_SIZE)
    )