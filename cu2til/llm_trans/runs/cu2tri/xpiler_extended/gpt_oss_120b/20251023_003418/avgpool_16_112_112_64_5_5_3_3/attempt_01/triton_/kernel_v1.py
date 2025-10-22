import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel (named exactly as required)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,                     # *float32, input tensor (NCHWc layout, blocked by 64)
    out_ptr,                   # *float32, output tensor (contiguous)
    batch_size,                # int32
    channels,                  # int32
    input_H,                   # int32
    kernel_size,               # int32 (expected 5)
    stride,                    # int32 (expected 1)
    output_H,                  # int32
    output_size,               # int64, total number of output elements
    BLOCK_SIZE: tl.constexpr,  # compile‑time constant (1024)
    C_BLOCK: tl.constexpr = 64,   # channel block size (blocked layout)
    KERNEL_SIZE: tl.constexpr = 5 # kernel size (5x5 average pool)
):
    pid = tl.program_id(0)
    # Linear offsets for this block
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE, dtype=tl.int64)
    mask = offs < output_size

    # ------------------------------------------------------------------
    # Decode output linear index into (n, c, h_out, w_out)
    # Output layout is contiguous: [batch, channel, height, width]
    # ------------------------------------------------------------------
    n = offs // (channels * output_H * output_H)
    rem = offs % (channels * output_H * output_H)
    c = rem // (output_H * output_H)
    rem2 = rem % (output_H * output_H)
    h_out = rem2 // output_H
    w_out = rem2 % output_H

    # ------------------------------------------------------------------
    # Compute blocked input address (top‑left corner of the 5×5 window)
    # Input layout: NCHWc, i.e.
    #   index = ((n * channel_blocks + c_block) * H * H + h * H + w) * C_BLOCK + c_offset
    # ------------------------------------------------------------------
    c_block = c // C_BLOCK
    c_offset = c % C_BLOCK
    channel_blocks = (channels + C_BLOCK - 1) // C_BLOCK

    # Base address for the top‑left element of the window
    base = ((n * channel_blocks + c_block) * input_H * input_H +
            h_out * input_H + w_out) * C_BLOCK + c_offset

    # Stride for moving one row down in the blocked layout
    stride_h = input_H * C_BLOCK  # = height_stride

    # ------------------------------------------------------------------
    # Accumulate sum over the 5×5 window
    # ------------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for rv0 in range(KERNEL_SIZE):
        off_h = rv0 * stride_h
        for rv1 in range(KERNEL_SIZE):
            addr = base + off_h + rv1 * C_BLOCK
            val = tl.load(A_ptr + addr, mask=mask, other=0.0)
            sum_val += val

    # ------------------------------------------------------------------
    # Compute average (1/25) and write back
    # ------------------------------------------------------------------
    avg = sum_val * (1.0 / (KERNEL_SIZE * KERNEL_SIZE))
    tl.store(out_ptr + offs, avg, mask=mask)


# ----------------------------------------------------------------------
# Wrapper entry point (exact signature as the original CUDA kernel)
# ----------------------------------------------------------------------
def triton_kernel(input, output, batch_size, channels, input_H, kernel_size, stride):
    """
    Triton implementation of a 5×5 average‑pooling kernel.
    Arguments:
        input (torch.Tensor):  float32 tensor on CUDA, shape (batch, channels, input_H, input_H)
                               stored in NCHWc layout (channel blocked by 64).
        output (torch.Tensor): float32 tensor on CUDA, shape (batch, channels, output_H, output_H),
                               contiguous layout.
        batch_size (int):      batch dimension.
        channels (int):        number of channels (must be a multiple of 64 for the blocked layout).
        input_H (int):         spatial height (and width) of the input.
        kernel_size (int):     size of the pooling kernel (expected 5).
        stride (int):          stride of the pooling (expected 1).
    """
    assert input.is_cuda and output.is_cuda, "input and output must be CUDA tensors"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "tensors must be float32"
    # Compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * channels * output_H * output_H

    # Kernel launch configuration
    BLOCK_SIZE = 1024
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        output,
        batch_size,
        channels,
        input_H,
        kernel_size,
        stride,
        output_H,
        output_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Ensure completion before returning
    torch.cuda.synchronize()