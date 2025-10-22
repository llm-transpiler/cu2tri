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
    """Per‑slice instance normalization.

    Each program instance (i.e. block) processes one (instance, channel) slice.
    """
    pid = tl.program_id(0)                     # slice id = instance * channels + channel
    channel = pid % channels
    slice_offset = pid * spatial                # start index of the slice in the flat tensor
    slice_end = slice_offset + spatial          # exclusive end index

    # Load per‑channel scale and shift
    scale = tl.load(gamma_ptr + channel)
    shift = tl.load(beta_ptr + channel)

    # ------------------------------------------------------------------
    # 1) Compute mean and variance of the slice
    # ------------------------------------------------------------------
    sum_val = 0.0
    sum_sq_val = 0.0
    offset = 0
    while offset < spatial:
        idx = slice_offset + offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < slice_end
        x = tl.load(input_ptr + idx, mask=mask, other=0.0)   # vector of size BLOCK_SIZE
        sum_val += tl.sum(x)                               # reduce across the block
        sum_sq_val += tl.sum(x * x)
        offset += BLOCK_SIZE

    inv_spatial = 1.0 / spatial
    mean = sum_val * inv_spatial
    var = sum_sq_val * inv_spatial - mean * mean
    eps = 1e-5
    inv_std = 1.0 / tl.sqrt(var + eps)

    # ------------------------------------------------------------------
    # 2) Normalize and write output
    # ------------------------------------------------------------------
    offset = 0
    while offset < spatial:
        idx = slice_offset + offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < slice_end
        x = tl.load(input_ptr + idx, mask=mask, other=0.0)
        normalized = (x - mean) * inv_std
        out = normalized * scale + shift
        tl.store(output_ptr + idx, out, mask=mask)
        offset += BLOCK_SIZE


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

    # Ensure contiguous layout for optimal performance
    input = input.contiguous()
    output = output.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    total_slices = batch * channels
    grid = (total_slices,)
    block = 256  # BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[grid, block](
        input,
        output,
        gamma,
        beta,
        batch,
        channels,
        spatial,
        BLOCK_SIZE=block,
    )
    # Synchronize to make sure the kernel has finished (useful for testing)
    torch.cuda.synchronize()