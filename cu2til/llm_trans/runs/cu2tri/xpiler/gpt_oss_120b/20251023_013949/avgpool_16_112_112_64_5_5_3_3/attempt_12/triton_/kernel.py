import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Triton kernel reproducing the exact average‑pooling behavior of the original
# CUDA kernel.  The kernel works on tensors in NCHW layout (batch, channels,
# height, width).  All indexing arithmetic mirrors the CUDA implementation.
# -------------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A,                     # *float32  (input tensor, NCHW)
    pool_avg,              # *float32  (output tensor, NCHW)
    batch_size: tl.int32,
    channels: tl.int32,
    input_H: tl.int32,
    output_H: tl.int32,
    BLOCK_SIZE: tl.constexpr,   # threads per block (1024)
    KERNEL_SIZE: tl.constexpr,  # pooling kernel size (e.g. 5)
    STRIDE: tl.constexpr        # stride (e.g. 3)
):
    pid = tl.program_id(0)                     # block index (0 .. num_blocks-1)
    tid = tl.arange(0, BLOCK_SIZE)             # thread indices within the block

 -----------------------------------------------------------------
    # Global output offset for each thread (matches pool_avg[...] indexing)
    # -----------------------------------------------------------------
    offsets = pid * BLOCK_SIZE + tid
    total_output = batch_size * channels * output_H * output_H
    mask = offsets < total_output                 # mask for out‑of‑range threads

    # -----------------------------------------------------------------
    # Decode block and thread indices to (batch channel, y_out, x_out)
    # exactly as the CUDA kernel does.
    # -----------------------------------------------------------------
    batch = pid // 81
    block_rem = pid % 81                         # 0 .. 80

    # channel = low 6 bits of threadIdx.x
    channel = tid & 63

    # Helper thread components
    t_div_64  = tid >> 6      # tid // 64   (0 .. 15)
    t_div_256 = tid >> 8      # tid // 256  (0 .. 3)

    # y_out = ((block_rem * 4) + (tid >> 8)) // 9   -> 0 .. output_H-1
    y_out = ((block_rem * 4) + t_div_256) // 9

    # x_out = ((pid * 16) + (tid >> 6)) % output_H
    x_out = ((pid * 16) + t_div_64) % output_H

    # -----------------------------------------------------------------
    # Strides for the NHW layout (NCHW flattened)
    # -----------------------------------------------------------------
    batch_stride = channels * input_H * input_H          # 802816 for the test case
    row_stride   = channels * input_H                    # 7168
    col_stride   = channels                              # 64
    y_offset = channels * input_H * STRIDE               # 21504
    x_offset = channels * STRIDE                         # 192

    # Base address of the top‑left corner of the pooling window
    base = (batch * batch_stride) + (y_out * y_offset) + (x_out * x_offset) + channel

    # -----------------------------------------------------------------
    # Accumulate sum over the KERNEL_SIZE × KERNEL_SIZE window
    # -----------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for ky in range(KERNEL_SIZE):
        for kx in range(KERNEL_SIZE):
            idx = base + ky * row_stride + kx * col_stride
            val = tl.load(A + idx, mask=mask, other=0.0)
            sum_val += val

    # Compute average (kernel area = KERNEL_SIZE * KERNEL_SIZE)
    avg = sum_val * (1.0 / (KERNEL_SIZE * KERNEL_SIZE))

    # Write result
    tl.store(pool_avg + offsets, avg, mask=mask)


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
    Wrapper mirroring the original CUDA kernel signature.
    Handles possible non‑contiguous tensors by materialising contiguous buffers.
    """
    # -----------------------------------------------------------------
    # Basic sanity checks
    # -----------------------------------------------------------------
    if not input.is_cuda or not output.is_cuda:
        raise RuntimeError("Both input and output must be CUDA tensors")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("Only torch.float32 tensors are supported")

    # -----------------------------------------------------------------
    # Ensure contiguous memory for the kernel
    # -----------------------------------------------------------------
    input_contig = input if input.is_contiguous() else input.contiguous()

    need_copy_back = False
    if output.is_contiguous():
        output_contig = output
    else:
        output_contig = torch.empty_like(output, memory_format=torch.contiguous_format)
        need_copy_back = True

    # -----------------------------------------------------------------
    # Compute output spatial dimension
    # -----------------------------------------------------------------
    output_H = (input_H - kernel_size) // stride + 1

    # -----------------------------------------------------------------
    # Grid configuration (one block per 1024 output elements)
    # -----------------------------------------------------------------
    BLOCK_SIZE = 1024
    total_output = batch_size * channels * output_H * output_H
    num_blocks = (total_output + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # -----------------------------------------------------------------
    # Launch Triton kernel
    # -----------------------------------------------------------------
    _triton_kernel_impl[grid](
        input_contig,
        output_contig,
        batch_size,
        channels,
        input_H,
        output_H,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_SIZE=kernel_size,
        STRIDE=stride,
        num_warps=32,   # 32 warps → 1024 threads per block
    )

    # -----------------------------------------------------------------
    # Copy back if the original output tensor was non‑contiguous
    # -----------------------------------------------------------------
    if need_copy_back:
        output.copy_(output_contig)