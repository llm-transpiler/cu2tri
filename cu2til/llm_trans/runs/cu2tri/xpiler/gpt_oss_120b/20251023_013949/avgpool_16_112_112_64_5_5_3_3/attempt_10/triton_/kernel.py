import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: average pooling over a K×K window (NHWC layout)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,                     # float* input tensor (NHWC)
    out_ptr,                   # float* output tensor (NHWC)
    batch_size: tl.int32,
    channels: tl.int32,
    input_H: tl.int32,
    output_H: tl.int32,
    stride: tl.int32,
    BLOCK_SIZE: tl.constexpr,  # threads per block (must match launch)
    KERNEL_SIZE: tl.constexpr, # pooling kernel size (e.g., 5)
):
    pid = tl.program_id(0)  # block index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # linear thread ids
    total_output = batch_size * output_H * output_H * channels
    mask = offs < total_output

    # Decode linear index into (n, h_out, w_out, c) for NHWC layout
    out_per_batch = output_H * output_H * channels
    n = offs // out_per_batch
    rem = offs % out_per_batch
    c = rem % channels
    hw = rem // channels
    h_out = hw // output_H
    w_out = hw % output_H

    # Base input coordinates for the top‑left corner of the kernel window
    h_in_base = h_out * stride
    w_in_base = w_out * stride

    # Accumulate sum over the K×K window
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(KERNEL_SIZE):
        h_in = h_in_base + i
        for j in range(KERNEL_SIZE):
            w_in = w_in_base + j
            # NHWC linear index: ((n * input_H + h_in) * input_H + w_in) * channels + c
            idx = ((n * input_H + h_in) * input_H + w_in) * channels + c
            sum_val += tl.load(A_ptr + idx, mask=mask, other=0.0)

    # Write average to output
    scale = 1.0 / (KERNEL_SIZE * KERNEL_SIZE)
    avg = sum_val * scale
    out_idx = ((n * output_H + h_out) * output_H + w_out) * channels + c
    tl.store(out_ptr + out_idx, avg, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(
    input_tensor: torch.Tensor,   # shape: (N, C, H, W) or (N, H, W, C)
    output_tensor: torch.Tensor,  # shape: (N, C, H_out, W_out) or (N, H_out, W_out, C)
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int,
):
    """
    Entry‑point that launches the Triton average‑pooling kernel.
    Mirrors the original CUDA kernel signature:
        cuda_kernel(float *input, float *output, int batch_size,
                    int channels, int input_H, int kernel_size, int stride)
    Supports both NCHW and NHWC layouts.
    """
    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------
    assert input_tensor.is_cuda and output_tensor.is_cuda, "Tensors must be CUDA tensors"
    assert input_tensor.dtype == torch.float32 and output_tensor.dtype == torch.float32, "Only float32 supported"
    assert input_tensor.dim() == 4 and output_tensor.dim() == 4, "Tensors must be 4‑D"
    assert input_tensor.shape[0] == batch_size, f"Batch size mismatch for input: {input_tensor.shape[0]} vs {batch_size}"
    assert output_tensor.shape[0] == batch_size, f"Batch size mismatch for output: {output_tensor.shape[0]} vs {batch_size}"

    # ------------------------------------------------------------------
    # Detect input layout and obtain NHWC view
    # ------------------------------------------------------------------
    if (
        input_tensor.shape[1] == channels
        and input_tensor.shape[2] == input_H
        and input_tensor.shape[3] == input_H
    ):
        # NCHW layout
        A_nhwc = input_tensor.permute(0, 2, 3, 1).contiguous()
    elif (
        input_tensor.shape[-1] == channels
        and input_tensor.shape[1] == input_H
        and input_tensor.shape[2] == input_H
    ):
        # NHWC layout
        A_nhwc = input_tensor.contiguous()
    else:
        raise AssertionError(f"Input tensor shape {input_tensor.shape} is not compatible with NCHW or NHWC layout")

    # ------------------------------------------------------------------
    # Compute output spatial dimension
    # ------------------------------------------------------------------
    output_H = (input_H - kernel_size) // stride + 1

    # ------------------------------------------------------------------
    # Detect output layout
    # ------------------------------------------------------------------
    if (
        output_tensor.shape[1] == channels
        and output_tensor.shape[2] == output_H
        and output_tensor.shape[3] == output_H
    ):
        # NCHW layout – allocate temporary NHWC buffer
        out_nhwc = torch.empty(
            batch_size,
            output_H,
            output_H,
            channels,
            dtype=output_tensor.dtype,
            device=output_tensor.device,
        )
        output_is_nhwc = False
    elif (
        output_tensor.shape[-1] == channels
        and output_tensor.shape[1] == output_H
        and output_tensor.shape[2] == output_H
    ):
        # NHWC layout – can write directly
        out_nhwc = output_tensor
        output_is_nhwc = True
    else:
        raise AssertionError(f"Output tensor shape {output_tensor.shape} is not compatible with NCHW or NHWC layout")

    # ------------------------------------------------------------------
    # Kernel launch configuration
    # ------------------------------------------------------------------
    BLOCK_SIZE = 1024
    total_output = batch_size * output_H * output_H * channels
    grid = ((total_output + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Launch Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A_nhwc,
        out_nhwc,
        batch_size,
        channels,
        input_H,
        output_H,
        stride,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_SIZE=kernel_size,
    )
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # If the original output was NCHW, copy the result back
    # ------------------------------------------------------------------
    if not output_is_nhwc:
        # out_nhwc has shape (N, H_out, W_out, C)
        output_tensor.copy_(out_nhwc.permute(0, 3, 1, 2))