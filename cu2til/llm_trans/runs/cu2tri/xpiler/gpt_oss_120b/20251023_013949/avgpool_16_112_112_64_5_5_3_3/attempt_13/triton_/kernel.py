import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: average pooling over a K×K window (valid padding, stride)
# Layout: NCHW (batch, channels, height, width)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # *float32, input tensor (NCHW, contiguous)
    pool_avg_ptr,        # *float32, output tensor (NCHW, contiguous)
    batch_size,          # int32
    channels,            # int32
    input_H,             # int32 (input height, square)
    stride,              # int32
    output_H,            # int32 (output height, square)
    total_output,        # int32 (batch * channels * output_H * output_H)
    BLOCK_SIZE: tl.constexpr,   # compile‑time block size (threads per program)
    KERNEL_SIZE: tl.constexpr,  # compile‑time kernel size (e.g. 5)
):
    """
    Each thread computes one output element:
        avg = sum_{i,j} A[n, c, h_out*stride+i, w_out*stride+j] / (K*K)
    """
    pid = tl.program_id(0)                     # block index
    offsets = tl.arange(0, BLOCK_SIZE)         # [0, 1, ..., BLOCK_SIZE‑1]
    out_idx = pid * BLOCK_SIZE + offsets       # linear output index

    # ------------------------------------------------------------------
    # Mask for threads that are out of the valid output range
    # ------------------------------------------------------------------
    mask = out_idx < total_output

    # ------------------------------------------------------------------
    # Decompose linear index into (n, c, h_out, w_out)
    # ------------------------------------------------------------------
    total_spatial = output_H * output_H
    nc_spatial = channels * total_spatial

    n = out_idx // nc_spatial
    rem = out_idx % nc_spatial
    c = rem // total_spatial
    rem2 = rem % total_spatial
    h_out = rem2 // output_H
    w_out = rem2 % output_H

    # ------------------------------------------------------------------
    # Top‑left corner of the kernel window in the input
    # ------------------------------------------------------------------
    h_start = h_out * stride
    w_start = w_out * stride

    # ------------------------------------------------------------------
    # Base offset of the top‑left element (NCHW layout)
    #   offset = ((n * C + c) * H + h_start) * W + w_start
    # ------------------------------------------------------------------
    base = ((n * channels + c) * input_H + h_start) * input_H + w_start

    # ------------------------------------------------------------------
    # Accumulate sum over the K×K window
    # ------------------------------------------------------------------
    sum_val = tl.zeros([1], dtype=tl.float32)   # scalar stored as 1‑element tensor
    for i in range(KERNEL_SIZE):
        row_offset = i * input_H
        for j in range(KERNEL_SIZE):
            idx = base + row_offset + j
            a = tl.load(A_ptr + idx, mask=mask, other=0.0)
            sum_val = sum_val + a                # broadcast addition, result shape [1]

    # ------------------------------------------------------------------
    # Compute average and write back
    # ------------------------------------------------------------------
    avg = sum_val[0] * (1.0 / (KERNEL_SIZE * KERNEL_SIZE))
    tl.store(pool_avg_ptr + out_idx, avg, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper that mimics the original CUDA entry point
# ----------------------------------------------------------------------
def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int,
) -> None:
    """
    Triton implementation of the original ``cuda_kernel``.
    Arguments must match the CUDA signature exactly.
    """
    # ------------------------------------------------------------------
    # Basic sanity checks (mirrors the expectations of the CUDA version)
    # ------------------------------------------------------------------
    if not isinstance(input, torch.Tensor) or not isinstance(output, torch.Tensor):
        raise TypeError("input and output must be torch.Tensor")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise TypeError("input and output must be torch.float32")
    if not input.is_cuda or not output.is_cuda:
        raise RuntimeError("input and output must be CUDA tensors")
    if input.ndim != 4 or output.ndim != 4:
        raise ValueError("input and output must be 4‑D tensors (NCHW)")

    # Ensure contiguous layout (required for pointer arithmetic)
    if not input.is_contiguous():
        input = input.contiguous()
    if not output.is_contiguous():
        output = output.contiguous()

    # Verify shapes
    if input.shape != (_size, channels, input_H, input_H):
        raise ValueError(
            f"input shape {input.shape} does not match "
            f"(batch_size={batch_size}, channels={channels}, input_H={input_H})"
        )
    if kernel_size <= 0 or stride <= 0:
        raise ValueError("kernel_size and stride must be positive integers")
    if input_H < kernel_size:
        raise ValueError("input_H must be >= kernel_size")

    # ------------------------------------------------------------------
    # Compute output spatial dimension (valid pooling, no padding)
    # ------------------------------------------------------------------
    output_H = (input_H - kernel_size) // stride + 1
    if output_H <= 0:
        raise ValueError("Computed output_H is non‑positive")
    expected_output_shape = (batch_size, channels, output_H, output_H)
    if output.shape != expected_output_shape:
        raise ValueError(
            f"output shape {output.shape} does not match expected {expected_output_shape}"
        )

    total_output = batch_size * channels * output_H * output_H

    # ------------------------------------------------------------------
    # Triton launch configuration
    # ------------------------------------------------------------------
    BLOCK_SIZE = 1024                     # matches the original CUDA launch bounds
    grid = ((total_output + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ---------------------------------------------------------------- # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        input,
        output,
        batch_size,
        channels,
        input_H,
        stride,
        output_H,
        total_output,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_SIZE=kernel_size,
        num_warps=32,                     # 32 warps == 1024 threads
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()