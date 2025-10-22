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
    NUM_ITERS: tl.constexpr,
):
    """
    One program instance (program_id) processes one (instance, channel) slice:
        slice = input[instance, channel, :]
    The kernel computes the slice mean/variance, normalizes the slice,
    and applies per‑channel scale (gamma) and shift (beta).
    """
    pid = tl.program_id(0)                     # 0‑D grid: pid ∈ [0, batch*channels)
    channel = pid % channels                    # channel index for this slice
    slice_offset = pid * spatial                # start index of the slice in the flat tensor
    slice_end = slice_offset + spatial          # exclusive end index

    # Load per‑channel affine parameters
    scale = tl.load(gamma_ptr + channel)
    shift = tl.load(beta_ptr + channel)

    # ------------------------------------------------------------------
    # 1) Compute mean and variance of the slice
    # ------------------------------------------------------------------
    # Initialize scalar accumulators (zero)
    sum_val = tl.zeros([1], dtype=tl.float32)[0]
    sum_sq_val = tl.zeros([1], dtype=tl.float32)[0]

    range_vec = tl.arange(0, BLOCK_SIZE)       # [BLOCK_SIZE] vector of offsets

    for i in range(NUM_ITERS):
        offset = i * BLOCK_SIZE
        idx = slice_offset + offset + range_vec
        mask = idx < slice_end
        x = tl.load(input_ptr + idx, mask=mask, other=0.0)   # load with zero‑fill for tail
        sum_val = sum_val + tl.sum(x)                       # accumulate sum
        sum_sq_val = sum_sq_val + tl.sum(x * x)             # accumulate sum of squares

    inv_spatial = 1.0 / spatial
    mean = sum_val * inv_spatial
    var = sum_sq_val * inv_spatial - mean * mean
    eps = 1e-5
    inv_std = 1.0 / tl.sqrt(var + eps)

    # ------------------------------------------------------------------
    # 2) Normalize the slice and write the result
    # ------------------------------------------------------------------
    for i in range(NUM_ITERS):
        offset = i * BLOCK_SIZE
        idx = slice_offset + offset + range_vec
        mask = idx < slice_end
        x = tl.load(input_ptr + idx, mask=mask, other=0.0)
        normalized = (x - mean) * inv_std
        out = normalized * scale + shift
        tl.store(output_ptr + idx, out, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper that mirrors the original CUDA kernel signature.
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
    Entry point with the exact same signature as the original CUDA kernel.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape (batch, channels, spatial), dtype torch.float32, CUDA.
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
    # ------------------------------------------------------------------
    # Basic sanity checks (mirroring the CUDA wrapper)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Grid configuration: one program instance per (instance, channel) slice
    # ------------------------------------------------------------------
    total_slices = batch * channels               # number of (instance, channel) pairs
    grid = total_slices                           # 1‑D grid (integer)
    block = 256                                    # BLOCK_SIZE (must match constexpr)

    # Number of iterations needed to cover the spatial dimension
    num_iters = (spatial + block - 1) // block     # ceil(spatial / BLOCK_SIZE)

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
        NUM_ITERS=num_iters,
    )
    # Synchronize for correctness when used in testing / benchmarking
    torch.cuda.synchronize()