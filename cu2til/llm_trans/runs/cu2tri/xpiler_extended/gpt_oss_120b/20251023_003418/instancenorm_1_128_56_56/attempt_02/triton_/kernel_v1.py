import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing instance normalization.
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    input_ptr,          # const float* __restrict__ input
    output_ptr,         # float* __restrict__ output
    gamma_ptr,          # const float* __restrict__ gamma
    beta_ptr,           # const float* __restrict__ beta
    batch: tl.int32,
    channels: tl.int32,
    spatial: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-element instance normalization.

    Each program instance processes BLOCK_SIZE consecutive elements of the
    flattened (batch, channels, spatial) tensor.
    """
    # ------------------------------------------------------------------
    # 1) Compute global linear index and mask out-of-range threads
    # ------------------------------------------------------------------
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE] vector
    total = batch * channels * spatial
    mask = idx < total

    # ------------------------------------------------------------------
    # 2) Derive (instance, channel) coordinates from the linear idx
    # ------------------------------------------------------------------
    # spatial dimension size
    sub = spatial
    channel = (idx // sub) % channels          # [BLOCK_SIZE] vector
    instance = idx // (channels * sub)          # [BLOCK_SIZE] vector

    # ------------------------------------------------------------------
    # 3) Pointer to the beginning of the (instance, channel) slice
    # ------------------------------------------------------------------
    slice_base = (instance * channels + channel) * spatial
    slice_ptr = input_ptr + slice_base

    # ------------------------------------------------------------------
    # 4) Compute mean of the slice (loop over spatial dimension)
    # ------------------------------------------------------------------
    sum_val = tl.float32(0.0)
    offset = tl.int32(0)
    while offset < spatial:
        cur_offset = offset + tl.arange(0, BLOCK_SIZE)               # [BLOCK_SIZE]
        mask_spatial = cur_offset < spatial
        vals = tl.load(slice_ptr + cur_offset, mask=mask_spatial, other=0.0)
        sum_val += tl.sum(vals)                                     # scalar
        offset += BLOCK_SIZE

    mean = sum_val / spatial

    # ------------------------------------------------------------------
    # 5) Compute variance of the slice
    # ------------------------------------------------------------------
    var_sum = tl.float32(0.0)
    offset = tl.int32(0)
    while offset < spatial:
        cur_offset = offset + tl.arange(0, BLOCK_SIZE)
        mask_spatial = cur_offset < spatial
        vals = tl.load(slice_ptr + cur_offset, mask=mask_spatial, other=0.0)
        diff = vals - mean
        var_sum += tl.sum(diff * diff)
        offset += BLOCK_SIZE

    var = var_sum / spatial

    # ------------------------------------------------------------------
    # 6) Load the original element, apply normalization, scale & shift
    # ------------------------------------------------------------------
    x = tl.load(input_ptr + idx, mask=mask, other=0.0)
    eps = tl.float32(1e-5)
    normalized = (x - mean) / tl.sqrt(var + eps)

    gamma_val = tl.load(gamma_ptr + channel, mask=mask, other=0.0)
    beta_val = tl.load(beta_ptr + channel, mask=mask, other=0.0)

    out = normalized * gamma_val + beta_val

    # ------------------------------------------------------------------
    # 7) Write back the result
    # ------------------------------------------------------------------
    tl.store(output_ptr + idx, out, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper matching the original CUDA kernel signature.
# ----------------------------------------------------------------------
def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    batch: int,
    channels: int,
    spatial: int,
) -> None:
    """
    Entry point that mimics the original CUDA kernel signature.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape (batch, channels, spatial) and dtype torch.float32.
    output : torch.Tensor
        Output tensor of the same shape as ``input``.
    gamma : torch.Tensor
        Scale parameters of shape (channels,).
    beta : torch.Tensor
        Shift parameters of shape (channels,).
    batch : int
        Number of instances in the batch.
    channels : int
        Number of channels per instance.
    spatial : int
        Spatial dimension size.
    """
    if not input.is_cuda or not output.is_cuda:
        raise RuntimeError("input and output tensors must be CUDA tensors")
    if not gamma.is_cuda or not beta.is_cuda:
        raise RuntimeError("gamma and beta tensors must be CUDA tensors")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("input and output must be float32 tensors")
    if gamma.dtype != torch.float32 or beta.dtype != torch.float32:
        raise RuntimeError("gamma and beta must be float32 tensors")

    # Ensure contiguous memory layout for optimal performance
    input = input.contiguous()
    output = output.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    total = batch * channels * spatial
    threads = 256  # BLOCK_SIZE
    blocks = (total + threads - 1) // threads

    # Launch the Triton kernel
    _triton_kernel_impl[blocks, threads](
        input,
        output,
        gamma,
        beta,
        batch,
        channels,
        spatial,
        BLOCK_SIZE=threads,
    )
    # Optional synchronization for debugging / correctness checks
    torch.cuda.synchronize()