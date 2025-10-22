import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel (named exactly as required)
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024  # matches the original CUDA block size

@triton.jit
def _triton_kernel_impl(
    A,                     # *float32 input tensor (flattened NCHW)
    pool_avg,              # *float32 output tensor (flattened NCHW)
    batch_size,            # int32
    channels,              # int32 (must be 64 for this mapping)
    input_H,               # int32
    kernel_size,           # int32 (fixed to 5 in the original kernel)
    stride,                # int32
    output_H,              # int32
    output_size,           # int32 (total number of output elements)
    BLOCK_SIZE: tl.constexpr  # compile‑time constant
):
    pid = tl.program_id(0)               # block index (equivalent to blockIdx.x)
    lane = tl.arange(0, BLOCK_SIZE)      # thread index within the block (threadIdx.x)

    # Global linear index for the output element handled by this thread
    out_idx = pid * BLOCK_SIZE + lane
    mask = out_idx < output_size          # guard for the last partially filled block

    # ------------------------------------------------------------------
    # Mapping from (pid, lane) to (batch, channel, out_y, out_x)
    # This reproduces the exact indexing logic of the CUDA kernel.
    # ------------------------------------------------------------------
    # Number of blocks that cover one batch (ceil division)
    blocks_per_batch = (channels * output_H * output_H + BLOCK_SIZE - 1) // BLOCK_SIZE

    batch = pid // blocks_per_batch
    subblock = pid % blocks_per_batch

    # The original kernel packs the channel in the low 6 bits of threadIdx.x
    # (channels = 64 → 6 bits). The remaining bits are used for spatial coords.
    channel = lane & (channels - 1)                 # low 6 bits
    highbits = lane >> (6 + 2)                      # bits 8‑9
    midbits = lane >> 6                              # bits 6‑9

    # Helper constants derived from the original magic numbers
    block_factor = BLOCK_SIZE // channels            # 16 for 1024/64
    factor_y = block_factor // 4                     # 4
    divisor_y = (blocks_per_batch * factor_y) // output_H  # 9 for the example

    # Output spatial coordinates
    out_y = ((subblock * factor_y + highbits) // divisor_y)
    out_x = ((pid * block_factor + midbits) % output_H)

    # ------------------------------------------------------------------
    # Compute the base offset into the input tensor (NCHW layout)
    # index = ((batch*C + channel)*H + (out_y*stride + rv0))*H + (out_x*stride + rv1)
    # ------------------------------------------------------------------
    batch_offset = batch * channels * input_H * input_H
    out_y_offset = out_y * channels * stride * input_H
    out_x_offset = out_x * channels * stride
    base = batch_offset + out_y_offset + out_x_offset + channel

    # ------------------------------------------------------------------
    # Accumulate the sum over the 5×5 window
    # ------------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # kernel_size is fixed to 5 in the original code; we unroll it explicitly.
    for rv0 in range(5):
        for rv1 in range(5):
            idx = base + rv0 * channels * input_H + rv1 * channels
            sum_val += tl.load(A + idx, mask=mask, other=0.0)

    # Write the sum to the output tensor
    tl.store(pool_avg + out_idx, sum_val, mask=mask)


# ----------------------------------------------------------------------
# Wrapper with the exact original signature
# ----------------------------------------------------------------------
def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int
):
    """
    Triton implementation of the original CUDA kernel.
    The argument list matches the CUDA entry point exactly.
    """
    # ------------------------------------------------------------------
    # Basic sanity checks (mirroring typical CUDA expectations)
    # ------------------------------------------------------------------
    assert input.is_cuda and output.is_cuda, "Input and output must be CUDA tensors"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Tensors must be float32"
    assert input.is_contiguous() and output.is_contiguous(), "Tensors must be contiguous"

    # Compute output spatial dimension (same formula as the CUDA host code)
    output_H = (input_H - kernel_size) // stride + 1
    assert output_H > 0, "Invalid output size computed"

    # Total number of output elements (flattened)
    output_size = batch_size * channels * output_H * output_H

    # Grid configuration: one program (block) per BLOCK_SIZE output elements
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        output,
        batch_size,
        channels,
        input_H,
        kernel_size,   # kept for signature compatibility; kernel assumes 5
        stride,
        output_H,
        output_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8    # 8 warps per block (256 threads); Triton will schedule 1024 threads
    )