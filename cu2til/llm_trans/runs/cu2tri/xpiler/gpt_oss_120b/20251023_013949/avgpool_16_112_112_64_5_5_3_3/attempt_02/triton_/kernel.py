import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Kernel configuration (matches the original CUDA launch bounds)
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024  # 1024 threads per block


@triton.jit
def _triton_kernel_impl(
    input_ptr,          # float* (NHWC layout, contiguous)
    output_ptr,         # float* (NHWC layout, contiguous)
    batch,              # int
    channels,           # int
    input_H,            # int (height == width)
    output_H,           # int (height == width)
    stride,             # int
    total_output_elems, # int (batch * output_H * output_H * channels)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton implementation of a 5×5 average‑pool with stride `stride`.
    Each thread produces one output element (batch, y, x, channel) in NHWC layout.
    """
    pid = tl.program_id(0)                     # block index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE, dtype=tl.int64)
    mask = offsets < total_output_elems

    # -------------------------------------------------------------
    # Decode linear offset -> (b, out_y, out_x, c) for NHWC layout
    # offset = ((b * output_H + out_y) * output_H + out_x) * channels + c
    # -------------------------------------------------------------
    c = offsets % channels
    idx = offsets // channels
    out_x = idx % output_H
    idx = idx // output_H
    out_y = idx % output_H
    b = idx // output_H

    # -------------------------------------------------------------
    # Top‑left corner of the pooling window in the input tensor
    # -------------------------------------------------------------
    in_y_base = out_y * stride
    in_x_base = out_x * stride

    # -------------------------------------------------------------
    # Accumulate the 5×5 sum
    # -------------------------------------------------------------
    sum_val = tl.float32(0.0)
    K = 5  # kernel size (fixed)

    for i in range(K):
        for j in range(K):
            in_y = in_y_base + i
            in_x = in_x_base + j
            # Input offset for NHWC layout:
            # ((b * input_H + in_y) * input_H + in_x) * channels + c
            input_offset = ((b * input_H + in_y) * input_H + in_x) * channels + c
            val = tl.load(
                input_ptr + input_offset,
                mask=mask,
                other=tl.float32(0.0),
            )
            sum_val += val

    # -------------------------------------------------------------
    # Compute average (1/25 = 0.04)
    # -------------------------------------------------------------
    avg = sum_val * (1.0 / (K * K))

    # -------------------------------------------------------------
    # Write result
    # -------------------------------------------------------------
    tl.store(output_ptr + offsets, avg, mask=mask)


def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int,
):
    """
    Wrapper that matches the original CUDA kernel signature.
    The original CUDA kernel expects NHWC layout; this wrapper accepts the
    conventional PyTorch NCHW layout, converts to NHWC for the Triton kernel,
    and converts the result back to NCHW.
    """
    # -----------------------------------------------------------------
    # Basic validation (mirrors expectations of the original CUDA code)
    # -----------------------------------------------------------------
    if not input.is_cuda or not output.is_cuda:
        raise RuntimeError("Input and output tensors must be CUDA tensors.")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("Only float32 tensors are supported    # The original implementation supports a 5×5 kernel.
    if kernel_size != 5:
        raise ValueError("This Triton implementation only supports kernel_size == 5.")

    # -----------------------------------------------------------------
    # Expected shapes (NCHW layout as used by the test harness)
    # -----------------------------------------------------------------
    output_H = (input_H - kernel_size) // stride + 1
    expected_input_shape = (batch_size, channels, input_H, input_H)
    expected_output_shape = (batch_size, channels, output_H, output_H)

    if input.shape != expected_input_shape:
        raise RuntimeError(
            f"Input shape {input.shape} does not match expected {expected_input_shape}."
        )
    if output.shape != expected_output_shape:
        raise RuntimeError(
            f"Output shape {output.shape} does not match expected {expected_output_shape}."
        )

    # -----------------------------------------------------------------
    # Convert NCHW -> NHWC (contiguous) for the Triton kernel
    # -----------------------------------------------------------------
    input_nhwc = input.permute(0, 2, 3, 1).contiguous()
    # Allocate a contiguous NHWC buffer for the output
    output_nhwc = torch.empty(
        (batch_size, output_H, output_H, channels),
        dtype=torch.float32,
        device=input.device,
    )

    # -----------------------------------------------------------------
    # Launch configuration
    # -----------------------------------------------------------------
    total_output_elems = batch_size * output_H * output_H * channels
    grid = ((total_output_elems + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # -----------------------------------------------------------------
    # Kernel launch
    # -----------------------------------------------------------------
    _triton_kernel_impl[grid](
        input_nhwc,
        output_nhwc,
        batch_size,
        channels,
        input_H,
        output_H,
        stride,
        total_output_elems,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.cuda.synchronize()

    # -----------------------------------------------------------------
    # Convert NHWC -> NCHW and copy back to the user‑provided output tensor
    # -----------------------------------------------------------------
    output.copy_(output_nhwc.permute(0, 3, 1, 2))