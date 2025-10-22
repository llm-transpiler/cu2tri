import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel (exact name required)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,                     # *float32, input tensor (blocked layout NCHWc)
    out_ptr,                   # *float32, output tensor (contiguous NCHW)
    batch_size,                # int32
    channels,                  # int32
    input_H,                   # int32
    stride,                    # int32 (expected 1)
    output_H,                  # int32
    output_size,               # int64, total number of output elements
    BLOCK_SIZE: tl.constexpr,  # compile‑time constant (1024)
    C_BLOCK: tl.constexpr = 64,   # channel block size (blocked layout)
    KERNEL_SIZE: tl.constexpr = 5 # kernel size (5×5 average pool)
):
    pid = tl.program_id(0)
    # Thread indices within the block
    tid = tl.arange(0, BLOCK_SIZE)                # int32
    # Global linear index for each thread
    offs = tl.cast(pid, tl.int64) * BLOCK_SIZE + tl.cast(tid, tl.int64)  # int64
    mask = offs < output_size

    # ------------------------------------------------------------------
    # Decode linear output index into (batch, channel, h_out, w_out)
    # ------------------------------------------------------------------
    elems_per_batch = tl.int64(channels) * tl.int64(output_H) * tl.int64(output_H)
    batch_idx = offs // elems_per_batch
    rem = offs % elems_per_batch
    channel_idx = rem // (tl.int64(output_H) * tl.int64(output_H))
    rem2 = rem % (tl.int64(output_H) * tl.int64(output_H))
    h_out = rem2 // tl.int64(output_H)
    w_out = rem2 % tl.int64(output_H)

    # ------------------------------------------------------------------
    # Channel block decomposition (blocked layout)
    # ------------------------------------------------------------------
    c_block = channel_idx // tl.int64(C_BLOCK)
    c_offset = channel_idx % tl.int64(C_BLOCK)
    channel_blocks = (channels + C_BLOCK - 1) // C_BLOCK
    channel_blocks = tl.int64(channel_blocks)

    # ------------------------------------------------------------------
    # Compute base address of the top‑left element of the 5×5 window
    # Input layout (blocked): ((batch * channel_blocks + c_block) * H * H
    #                         + h_in * H + w_in) * C_BLOCK + c_offset
    # where h_in = h_out * stride, w_in = w_out * stride
    # ------------------------------------------------------------------
    h_in = h_out * tl.int64(stride)
    w_in = w_out * tl.int64(stride)
    base = ((batch_idx * channel_blocks + c_block) * tl.int64(input_H) * tl.int64(input_H)
            + h_in * tl.int64(input_H) + w_in) * tl.int64(C_BLOCK) + c_offset

    # ------------------------------------------------------------------
    # Accumulate sum over the 5×5 window
    # ------------------------------------------------------------------
    sum_val = tl.float32(0.0)
    for rv0 in range(KERNEL_SIZE):
        row_offset = rv0 * tl.int64(input_H) * tl.int64(C_BLOCK)
        for rv1 in range(KERNEL_SIZE):
            col_offset = rv1 * tl.int64(C_BLOCK)
            addr = base + row_offset + col_offset
            val = tl.load(A_ptr + addr, mask=mask, other=tl.float32(0.0))
            sum_val += val

    # ------------------------------------------------------------------
    # Compute average (1/25) and write back
    # ------------------------------------------------------------------
    avg = sum_val * (1.0 / (KERNEL_SIZE * KERNEL_SIZE))
    tl.store(out_ptr + offs, avg, mask=mask)


# ----------------------------------------------------------------------
# Wrapper entry point (identical signature to the original CUDA kernel)
# ----------------------------------------------------------------------
def triton_kernel(input, output, batch_size, channels, input_H, kernel_size, stride):
    """
    Triton implementation of the 5×5 average‑pooling kernel.
    Arguments:
        input (torch.Tensor):  float32 tensor on CUDA, shape (batch, channel_blocks, input_H, input_H, 64)
                               (blocked NCHWc layout, C_BLOCK = 64).
        output (torch.Tensor): float32 tensor on CUDA, shape (batch, channels, output_H, output_H),
                               contiguous layout.
        batch_size (int):      batch dimension.
        channels (int):        number of channels (must be a multiple of 64).
        input_H (int):         spatial height (and width) of the input.
        kernel_size (int):     size of the pooling kernel (expected 5).
        stride (int):          stride of the pooling (expected 1).
    """
    assert input.is_cuda and output.is_cuda, "input and output must be CUDA tensors"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "tensors must be float32"
    assert kernel_size == 5, "Only kernel size 5 is supported"
    assert stride == 1, "Only stride 1 is supported"
    assert channels % 64 == 0, "Number of channels must be a multiple of 64 (C_BLOCK)"

    # Compute output spatial dimension and total size
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
        stride,
        output_H,
        output_size,
        BLOCK_SIZE=BLOCK_SIZE,
        C_BLOCK=64,
        KERNEL_SIZE=kernel_size,
    )
    # Synchronize to make sure kernel has finished
    torch.cuda.synchronize()