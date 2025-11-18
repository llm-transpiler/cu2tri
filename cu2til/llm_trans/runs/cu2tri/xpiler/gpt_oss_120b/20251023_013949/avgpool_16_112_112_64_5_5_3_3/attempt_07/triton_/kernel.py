import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel (must be named exactly as required)
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024  # matches the CUDA launch bounds

@triton.jit
def _triton_kernel_impl(
    A_ptr, out_ptr,
    batch_size, channels,
    input_H, input_W,
    output_H, output_W,
    stride,
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
):
    """
    Average‑pooling kernel for NHWC layout.
    Computes:
        out[b, y, x, c] = mean(A[b, y*stride+ky, x*stride+kx, c])
    over a KERNEL_SIZE × KERNEL_SIZE window.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    idx = pid * BLOCK_SIZE + offs

    total_output = batch_size * output_H * output_W * channels
    mask = idx < total_output

    # ------------------------------------------------------------------
    # Decode linear index -> (b, y, x, c) for NHWC layout
    # ------------------------------------------------------------------
    c = idx % channels
    tmp = idx // channels
    x = tmp % output_W
    tmp2 = tmp // output_W
    y = tmp2 % output_H
    b = tmp2 // output_H

    # Top‑left corner of the pooling window in the input tensor
    in_y = y * stride
    in_x = x * stride

    # Base offset for the input element (NHWC layout)
    # ((b * H + in_y) * W + in_x) * C + c
    input_offset = ((b * input_H + in_y) * input_W + in_x) * channels + c

    # ------------------------------------------------------------------
    # Accumulate sum over the KERNEL_SIZE × KERNEL_SIZE window
    # ------------------------------------------------------------------
    sum_val = tl.float32(0.0)
    for ky in range(KERNEL_SIZE):
        for kx in range(KERNEL_SIZE):
            offset = input_offset + (ky * input_W + kx) * channels
            sum_val += tl.load(A_ptr + offset, mask=mask, other=0.0)

    # Compute the average
    avg = sum_val * (1.0 / (KERNEL_SIZE * KERNEL_SIZE))

    # Write the result (masked)
    tl.store(out_ptr + idx, avg, mask=mask)


# ----------------------------------------------------------------------
# Wrapper entry point (identical signature to the original CUDA wrapper)
# ----------------------------------------------------------------------
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
    Triton implementation of the average‑pooling kernel.
    Mirrors the original CUDA kernel signature.
    Handles non‑contiguous tensors by making temporary contiguous copies.
    """
    # ------------------------------------------------------------------
    # Basic sanity checks
    # ------------------------------------------------------------------
    assert input.is_cuda and output.is_cuda, "Tensors must be on a CUDA device"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 is supported"

    # Ensure contiguous memory for Triton (copy only when necessary)
    input_contig = input if input.is_contiguous() else input.contiguous()
    output_contig = output if output.is_contiguous() else output.contiguous()

    # ------------------------------------------------------------------
    # Compute output spatial dimensions (square case, matches original code)
    # ------------------------------------------------------------------
    output_H = (input_H - kernel_size) // stride + 1
    output_W = output_H  # square output

    total_output = batch_size * output_H * output_W * channels
    grid = ((total_output + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        input_contig,
        output_contig,
        batch_size,
        channels,
        input_H,
        input_H,          # input_W (square)
        output_H,
        output_W,
        stride,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_SIZE=kernel_size,
    )

    # If the original output tensor was non‑contiguous, copy the result back
    if not output.is_contiguous():
        output.copy_(output_contig)


# ----------------------------------------------------------------------
# Simple sanity‑check (executed when the file is run)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example configuration matching the test harness
    batch = 16
    channels = 64
    input_H = 112
    kernel = 5
    stride = 3

    # Derived output dimensions
    output_H = (input_H - kernel) // stride + 1
    out_shape = (batch, output_H, output_H, channels)  # NHWC

    # Allocate input / output tensors (NHWC layout)
    x_nchw = torch.randn(batch, channels, input_H, input_H, device="cuda", dtype=torch.float32)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()   # NHWC
    y_nhwc = torch.empty(out_shape, device="cuda", dtype=torch.float32)

    # Run Triton implementation
    triton_kernel(x_nhwc, y_nhwc, batch, channels, input_H, kernel, stride)
    torch.cuda.synchronize()

    # Reference using PyTorch avg_pool2d (NCHW layout)
    y_ref_nchw = torch.nn.functional.avg_pool2d(
        x_nchw,
        kernel_size=kernel,
        stride=stride,
        divisor_override=kernel * kernel,
    )
    y_ref_nhwc = y_ref_nchw.permute(0, 2, 3, 1)  # back to NHWC

    # Verify correctness
    max_err = (y_nhwc - y_ref_nhwc).abs().max().item()
    print(f"Max absolute error vs. torch.nn.functional.avg_pool2d: {max_err:e}")