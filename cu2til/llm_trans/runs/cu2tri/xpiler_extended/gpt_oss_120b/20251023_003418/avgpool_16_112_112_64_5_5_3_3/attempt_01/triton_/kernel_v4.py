import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel (must be named exactly as required)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,                     # *float32, input tensor (blocked layout NCHWc)
    out_ptr,                   # *float32, output tensor (contiguous NCHW)
    batch_size,                # int32
    channels,                  # int32
    input_H,                   # int32
    kernel_size,               # int32 (runtime, but KERNEL_SIZE is constexpr)
    stride,                    # int32
    output_H,                  # int32
    output_size,               # int64, total number of output elements
    BLOCK_SIZE: tl.constexpr,  # compile‑time constant (1024)
    C_BLOCK: tl.constexpr = 64,   # channel block size (must match input layout)
    KERNEL_SIZE: tl.constexpr = 5 # kernel size (compile‑time constant)
):
    pid = tl.program_id(0)

    # Linear offsets for this block (int64)
    offs = tl.int64(pid) * tl.int64(BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE, dtype=tl.int32)
    offs = tl.int64(offs)  # ensure 64‑bit for address arithmetic
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
    # Compute input top‑left corner of the K×K window
    # ------------------------------------------------------------------
    h_in = h_out * tl.int64(stride)
    w_in = w_out * tl.int64(stride)

    # ------------------------------------------------------------------
    # Channel block decomposition (blocked layout NCHWc)
    # ------------------------------------------------------------------
    c_block = channel_idx // tl.int64(C_BLOCK)
    c_offset = channel_idx % tl.int64(C_BLOCK)
    channel_blocks = (channels + C_BLOCK - 1) // C_BLOCK
    channel_blocks = tl.int64(channel_blocks)

    # Base address for the top‑left element of the window
    # ((batch * channel_blocks + c_block) * H * H + (h_in * H + w_in)) * C_BLOCK + c_offset
    base = ((batch_idx * channel_blocks + c_block) * tl.int64(input_H) * tl.int64(input_H)
            + h_in * tl.int64(input_H) + w_in) * tl.int64(C_BLOCK) + c_offset

    # ------------------------------------------------------------------
    # Accumulate sum over the K×K window
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
    # Compute average and write back
    # ------------------------------------------------------------------
    inv_area = 1.0 / (KERNEL_SIZE * KERNEL_SIZE)
    avg = sum_val * inv_area
    tl.store(out_ptr + offs, avg, mask=mask)


# ----------------------------------------------------------------------
# Wrapper entry point (identical signature to the original CUDA kernel)
# ----------------------------------------------------------------------
def triton_kernel(input, output, batch_size, channels, input_H, kernel_size, stride):
    """
    Triton implementation of the average‑pooling kernel.
    Arguments:
        input (torch.Tensor):  float32 tensor on CUDA, shape (batch, channels, input_H, input_H)
                               stored in blocked NCHWc layout with channel block size 64.
        output (torch.Tensor): float32 tensor on CUDA, shape (batch, channels, output_H, output_H),
                               contiguous layout.
        batch_size (int):      batch dimension.
        channels (int):        number of channels (must be a multiple of 64).
        input_H (int):         spatial height (and width) of the input.
        kernel_size (int):     size of the pooling kernel (expected 5).
        stride (int):          stride of the pooling (e.g., 3).
    """
    assert input.is_cuda and output.is_cuda, "input and output must be CUDA tensors"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "tensors must be float32"
    assert channels % 64 == 0, "Number of channels must be a multiple of 64 (C_BLOCK=64)"
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
        C_BLOCK=64,
        KERNEL_SIZE=kernel_size,   # compile‑time constant (kernel_size == 5 in tests)
    )
    # Synchronize to guarantee completion
    torch.cuda.synchronize()