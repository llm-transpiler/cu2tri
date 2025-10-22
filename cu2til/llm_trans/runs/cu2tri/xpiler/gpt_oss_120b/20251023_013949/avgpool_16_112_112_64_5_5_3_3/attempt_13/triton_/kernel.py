import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: average pooling over a K×K window (default K=5)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # *float32, input feature map (NCHW)
    pool_avg_ptr,        # *float32, output feature map (NCHW)
    batch_size,          # int32
    channels,            # int32
    input_H,             # int32
    stride,              # int32
    output_H,            # int32
    BLOCK_SIZE: tl.constexpr,   # compile‑time block size (threads per program)
    KERNEL_SIZE: tl.constexpr,  # compile‑time kernel size (e.g. 5)
):
    """
    Compute average pooling for a single output element per thread.
    The kernel assumes NCHW layout and a stride that may be >1.
    """
    # ------------------------------------------------------------------
    # 1) Compute a flat output index for each thread
    # ------------------------------------------------------------------
    pid = tl.program_id(0)                     # block index
    offsets = tl.arange(0, BLOCK_SIZE)         # [0, 1, ..., BLOCK_SIZE-1]
    out_idx = pid * BLOCK_SIZE + offsets       # global linear index

    # Total number of output elements (N * C * H_out * W_out)
    total_output = batch_size * channels * output_H * output_H
    mask = out_idx < total_output               # mask for out‑of‑range threads

    # ------------------------------------------------------------------
    # 2) De‑compose the flat index into (n, c, h_out, w_out)
    #    out_idx = ((n * C + c) * H_out + h_out) * H_out + w_out
    # ------------------------------------------------------------------
    spatial = channels * output_H * output_H
    n = out_idx // spatial
    rem = out_idx % spatial
    c = rem // (output_H * output_H)
    rem2 = rem % (output_H * output_H)
    h_out = rem2 // output_H
    w_out = rem2 % output_H

    # ------------------------------------------------------------------
    # 3) Compute the base offset of the top‑left element of the K×K window
    # ------------------------------------------------------------------
    # batch‑channel index
    nc = n * channels + c
    # top‑left corner in the input tensor (row‑major NCHW)
    # input index = ((nc * input_H + (h_out * stride)) * input_H) + (w_out * stride)
    h_base = h_out * stride
    w_base = w_out * stride
    input_base = ((nc * input_H + h_base) * input_H) + w_base

    # ------------------------------------------------------------------
    # 4) Accumulate the sum over the K×K window
    # ------------------------------------------------------------------
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(KERNEL_SIZE):
        row_offset = i * input_H
        for j in range(KERNEL_SIZE):
            col_offset = j
            a_idx = input_base + row_offset + col_offset
            a_val = tl.load(A_ptr + a_idx, mask=mask, other=0.0)
            acc += a_val

    # ------------------------------------------------------------------
    # 5) Compute the average and write back
    # ------------------------------------------------------------------
    avg = acc * (1.0 / (KERNEL_SIZE * KERNEL_SIZE))
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
    if input.shape != (batch_size, channels, input_H, input_H):
        raise ValueError(
            f"input shape {input.shape} does not match "
            f"(batch_size={batch_size}, channels={channels}, input_H={input_H})"
        )
    if kernel_size <= 0 or stride <= 0:
        raise ValueError("kernel_size and stride must be positive integers")
    if input_H < kernel_size:
        raise ValueError("input_H must be >= kernel_size")

    # ------------------------------------------------------------------
    # Compute output spatial dimension and total size
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

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        input,
        output,
        batch_size,
        channels,
        input_H,
        stride,
        output_H,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_SIZE=kernel_size,
        num_warps=32,                     # 32 warps == 1024 threads
    )
    # Ensure completion before returning to the caller
    torch.cuda.synchronize()