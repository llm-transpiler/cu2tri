import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr,   # const float*
    output_ptr,  # float*
    gamma_ptr,   # const float*
    beta_ptr,    # const float*
    batch: tl.int32,
    channels: tl.int32,
    spatial: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (batch, channel) slice
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels

    # Base offset of the slice (batch, channel, spatial)
    slice_offset = (batch_idx * channels + channel_idx) * spatial

    # Thread-local offsets within the slice
    offs = tl.arange(0, BLOCK_SIZE)  # int32 offsets

    # -----------------------------------------------------------------
    # First pass: compute sum and sum of squares over the spatial dim
    # -----------------------------------------------------------------
    partial_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    partial_sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    offset = 0
    while offset < spatial:
        cur_offs = offset + offs
        mask = cur_offs < spatial
        x = tl.load(input_ptr + slice_offset + cur_offs, mask=mask, other=0.0)
        partial_sum += tl.where(mask, x, 0.0)
        partial_sum_sq += tl.where(mask, x * x, 0.0)
        offset += BLOCK_SIZE

    # Reduce across the block to obtain totals for this slice
    total_sum = tl.sum(partial_sum)
    total_sum_sq = tl.sum(partial_sum_sq)

    # -----------------------------------------------------------------
    # Compute mean, variance, and the normalization factor
    # -----------------------------------------------------------------
    spatial_f = tl.cast(spatial, tl.float32)
    mean = total_sum / spatial_f
    var = total_sum_sq / spatial_f - mean * mean
    var = tl.maximum(var, 0.0)                     # guard against tiny negatives
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    # Load per‑channel scale (gamma) and shift (beta)
    scale = tl.load(gamma_ptr + channel_idx)
    shift = tl.load(beta_ptr + channel_idx)

    # -----------------------------------------------------------------
    # Second pass: write normalized output
    # -----------------------------------------------------------------
    offset = 0
    while offset < spatial:
        cur_offs = offset + offs
        mask = cur_offs < spatial
        x = tl.load(input_ptr + slice_offset + cur_offs, mask=mask, other=0.0)
        normalized = (x - mean) * inv_std
        y = normalized * scale + shift
        tl.store(output_ptr + slice_offset + cur_offs, y, mask=mask)
        offset += BLOCK_SIZE


def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    batch: int,
    channels: int,
    spatial: int,
):
    """
    Triton implementation of the instance normalization kernel.
    Mirrors the signature of the original CUDA entry point.
    """
    # -----------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------
    if not (input.is_cuda and output.is_cuda and gamma.is_cuda and beta.is_cuda):
        raise RuntimeError("All tensors must be on a CUDA device")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("Input and output tensors must be float32")
    if gamma.dtype != torch.float32 or beta.dtype != torch.float32:
        raise RuntimeError("Gamma and beta tensors must be float32")

    # Ensure contiguous layout (required for pointer arithmetic)
    input = input.contiguous()
    output = output.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    BLOCK_SIZE = 256
    # One program per (batch, channel) slice
    grid = (batch * channels,)

    _triton_kernel_impl[grid](
        input,
        output,
        gamma,
        beta,
        batch,
        channels,
        spatial,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,   # 256 threads = 8 warps
    )