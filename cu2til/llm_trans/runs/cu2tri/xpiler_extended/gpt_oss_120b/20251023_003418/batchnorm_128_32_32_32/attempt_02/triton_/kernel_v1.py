import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BLOCK_SIZE = 256          # threads per program (matches CUDA launch)
EPS = 1e-5                # epsilon for numerical stability

# ----------------------------------------------------------------------
# Triton kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    input_ptr,          # *float32
    output_ptr,         # *float32
    mean_ptr,           # *float32
    var_ptr,            # *float32
    gamma_ptr,          # *float32
    beta_ptr,           # *float32
    batch_size: tl.int32,
    num_channels: tl.int32,
    spatial_size: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    """
    Implements:
        output[idx] = ((input[idx] - mean[c]) / sqrt(var[c] + EPS)) * gamma[c] + beta[c]
    where c = (idx // spatial_size) % num_channels
    """
    pid = tl.program_id(0)                     # program (block) index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # linear indices for this block

    total = batch_size * num_channels * spatial_size
    mask = offsets < total                     # guard against out‑of‑bounds threads

    # Load input element
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Determine channel for each element
    channel = (offsets // spatial_size) % num_channels

    # Load per‑channel parameters
    mean_val  = tl.load(mean_ptr  + channel, mask=mask, other=0.0)
    var_val   = tl.load(var_ptr   + channel, mask=mask, other=0.0)
    gamma_val = tl.load(gamma_ptr + channel, mask=mask, other=0.0)
    beta_val  = tl.load(beta_ptr  + channel, mask=mask, other=0.0)

    # Batch‑norm computation
    norm = (x - mean_val) / tl.sqrt(var_val + EPS)
    out = norm * gamma_val + beta_val

    # Write result
    tl.store(output_ptr + offsets, out, mask=mask)


# ----------------------------------------------------------------------
# Wrapper (entry point) – matches the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    batch_size: int,
    num_channels: int,
    spatial_size: int
):
    """
    Triton implementation of the CUDA batchnorm kernel.
    Parameters have the same order and meaning as the original CUDA kernel.
    """
    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    assert input.is_cuda and output.is_cuda, "input and output must be CUDA tensors"
    assert mean.is_cuda and var.is_cuda and gamma.is_cuda and beta.is_cuda, \
        "mean, var, gamma, beta must be CUDA tensors"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, \
        "input and output must be float32"
    assert mean.dtype == var.dtype == gamma.dtype == beta.dtype == torch.float32, \
        "mean, var, gamma, beta must be float32"

    # Ensure contiguous layout (required for pointer arithmetic)
    input = input.contiguous()
    output = output.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # Flatten input and output for linear indexing (same as CUDA view)
    input_flat = input.view(-1)
    output_flat = output.view(-1)

    total = batch_size * num_channels * spatial_size
    grid = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    # ------------------------------------------------------------------
    # Launch Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        input_flat,
        output_flat,
        mean,
        var,
        gamma,
        beta,
        batch_size,
        num_channels,
        spatial_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    # Optional synchronization (useful for profiling / correctness checks)
    torch.cuda.synchronize()