import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: computes average pooling over a KxK window (K = KERNEL_SIZE)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,                     # float* input tensor (NHWC layout)
    out_ptr,                   # float* output tensor (NHWC layout)
    batch_size,                # int
    channels,                  # int
    input_H,                   # int (assumed square: input_W = input_H)
    output_H,                  # int (assumed square: output_W = output_H)
    stride,                    # int
    BLOCK_SIZE: tl.constexpr,  # int, threads per block (must match launch)
    KERNEL_SIZE: tl.constexpr, # int, pooling kernel size (e.g., 5)
):
    # ------------------------------------------------------------------
    # Compute a linear index for each thread
    # ------------------------------------------------------------------
    pid = tl.program_id(0)                     # block index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread linear ids
    total_output = batch_size * output_H * output_H * channels
    mask = offs < total_output

    # ------------------------------------------------------------------
    # Decode linear index into (n, h_out, w_out, c)
    # ------------------------------------------------------------------
    out_per_batch = output_H * output_H * channels
    n = offs // out_per_batch
    rem = offs % out_per_batch
    c = rem % channels
    hw = rem // channels
    h_out = hw // output_H
    w_out = hw % output_H

    # ------------------------------------------------------------------
    # Compute input coordinates for the top‑left corner of the KxK window
    # ------------------------------------------------------------------
    h_in_base = h_out * stride
    w_in_base = w_out * stride

    # ------------------------------------------------------------------
    # Accumulate sum over the KxK window
    # ------------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(KERNEL_SIZE):
        h_in = h_in_base + i
        for j in range(KERNEL_SIZE):
            w_in = w_in_base + j
            # Flattened NHWC index: ((n * H + h) * W + w) * C + c
            idx = ((n * input_H + h_in) * input_H + w_in) * channels + c
            sum_val += tl.load(A_ptr + idx, mask=mask, other=0.0)

    # ------------------------------------------------------------------
    # Write average to output
    # ------------------------------------------------------------------
    scale = 1.0 / (KERNEL_SIZE * KERNEL_SIZE)
    avg = sum_val * scale
    out_idx = ((n * output_H + h_out) * output_H + w_out) * channels + c
    tl.store(out_ptr + out_idx, avg, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(
    input_tensor: torch.Tensor,   # shape: (batch, H, H, channels)
    output_tensor: torch.Tensor,  # shape: (batch, H_out, H_out, channels)
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int,
):
    """
    Entry‑point that launches the Triton average‑pooling kernel.
    The signature mirrors the original CUDA kernel:
        cuda_kernel(float *input, float *output, int batch_size,
                    int channels, int input_H, int kernel_size, int stride)
    """
    assert input_tensor.is_cuda and output_tensor.is_cuda, "Tensors must be on CUDA"
    assert input_tensor.dtype == torch.float32 and output_tensor.dtype == torch.float32
    # Derive output spatial size
    output_H = (input_H - kernel_size) // stride + 1
    # Ensure output tensor has the expected shape
    expected_out_shape = (batch_size, output_H, output_H, channels)
    assert output_tensor.shape == expected_out_shape, f"output shape mismatch, expected {expected_out_shape}"
    # Flatten tensors (NHWC layout) – Triton works with raw pointers
    A = input_tensor
    B = output_tensor

    # Kernel launch configuration
    BLOCK_SIZE = 1024
    total_output = batch_size * output_H * output_H * channels
    grid = (total_output + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A,
        B,
        batch_size,
        channels,
        input_H,
        output_H,
        stride,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_SIZE=kernel_size,
    )
    # Synchronize to make the kernel effect visible to the host
    torch.cuda.synchronize()