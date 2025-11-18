import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel that reproduces the original CUDA average‑pooling behavior
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,                # float* (input tensor, flattened NCHW)
    pool_avg_ptr,         # float* (output tensor, flattened NCHW)
    batch_size,           # int64
    channels,             # int64
    input_H,              # int64 (input height, square)
    stride,               # int64
    output_H,             # int64 (output height)
    output_size,          # int64 (total number of output elements)
    kernel_size: tl.constexpr,   # compile‑time kernel size (e.g. 5)
    BLOCK_SIZE: tl.constexpr,    # compile‑time block size (must be 1024)
):
    # ------------------------------------------------------------------
    # Thread / block identifiers (equivalent to CUDA's blockIdx.x / threadIdx.x)
    # ------------------------------------------------------------------
    pid = tl.program_id(0)                     # blockIdx.x (int32)
    pid_i64 = tl.cast(pid, tl.int64)

    tid = tl.arange(0, BLOCK_SIZE)             # threadIdx.x vector (int32)
    tid_i64 = tl.cast(tid, tl.int64)

    # Linear index of the output element processed by each lane
    out_idx = pid_i64 * BLOCK_SIZE + tid_i64
    mask = out_idx < output_size                # guard against out‑of‑bounds lanes

    # ------------------------------------------------------------------
    # Pre‑compute constants that are independent of the per‑lane mask
    # ------------------------------------------------------------------
    out_hw = output_H * output_H                # elements per channel in output
    channel_stride = input_H * input_H          # H * W per channel
    batch_stride = channels * channel_stride    # C * H * W per batch

    # ------------------------------------------------------------------
    # Compute only for active lanes to avoid out‑of‑bounds address calc
    # ------------------------------------------------------------------
    if mask:
        # Decode (batch, channel, out_y, out_x) from the flat output index
        n_c = out_idx // out_hw                 # (batch * channels) index
        rem = out_idx % out_hw
        out_y = rem // output_H
        out_x = rem % output_H
        n = n_c // channels                     # batch index
        c = n_c % channels                      # channel index

        # ------------------------------------------------------------------
        # Accumulate sum over the K×K pooling window
        # ------------------------------------------------------------------
        sum_val = tl.cast(0, tl.float32)        # scalar accumulator

        for rv0 in range(kernel_size):
            for rv1 in range(kernel_size):
                in_y = out_y * stride + rv0
                in_x = out_x * stride + rv1
                input_idx = (
                    n * batch_stride
                    + c * channel_stride
                    + in_y * input_H
                    + in_x
                )
                a = tl.load(A_ptr + input_idx, other=0.0)
                sum_val = sum_val + a

        # ------------------------------------------------------------------
        # Write the average (sum * 1/(K*K)) to the output tensor
        # ------------------------------------------------------------------
        avg = sum_val * (1.0 / (kernel_size * kernel_size))
        tl.store(pool_avg_ptr + out_idx, avg)


# ----------------------------------------------------------------------
# Wrapper matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(
    input: torch.Tensor,   # float* input tensor (NCHW)
    output: torch.Tensor,  # float* output tensor (NCHW)
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int,
):
    """
    Launches the Triton average‑pooling kernel.
    The semantics are identical to the original CUDA kernel.
    """
    # Compute output spatial dimension and total number of output elements
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * channels * output_H * output_H

    BLOCK_SIZE = 1024
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Ensure tensors are on CUDA and contiguous
    if not input.is_cuda:
        input = input.cuda()
    if not output.is_cuda:
        output = output.cuda()
    input = input.contiguous()
    output = output.contiguous()

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
        kernel_size=kernel_size,   # constexpr
        BLOCK_SIZE=BLOCK_SIZE,     # constexpr
    )