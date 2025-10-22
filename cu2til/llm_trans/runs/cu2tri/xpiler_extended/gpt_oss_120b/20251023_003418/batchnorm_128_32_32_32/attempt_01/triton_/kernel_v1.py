import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr, output_ptr,
    mean_ptr, var_ptr,
    gamma_ptr, beta_ptr,
    batch_size, num_channels, spatial_size,
    total,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = pid * BLOCK_SIZE + offsets
    mask = idx < total

    # Compute per-element channel index: (idx // spatial_size) % num_channels
    channel = tl.mod(tl.div(idx, spatial_size), num_channels)

    # Load values (masked loads to avoid OOB)
    x = tl.load(input_ptr + idx, mask=mask, other=0.0)
    m = tl.load(mean_ptr + channel, mask=mask, other=0.0)
    v = tl.load(var_ptr + channel, mask=mask, other=0.0)
    g = tl.load(gamma_ptr + channel, mask=mask, other=0.0)
    b = tl.load(beta_ptr + channel, mask=mask, other=0.0)

    # BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
    eps = 1e-5
    norm = (x - m) / tl.sqrt(v + eps)
    y = norm * g + b

    tl.store(output_ptr + idx, y, mask=mask)

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
    Triton implementation of the original CUDA batchnorm kernel.
    Argument order and types match the CUDA kernel signature.
    """
    # Basic sanity checks
    assert input.is_cuda and output.is_cuda and mean.is_cuda and var.is_cuda and gamma.is_cuda and beta.is_cuda, \
        "All tensors must reside on the CUDA device"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, \
        "Input and output tensors must be float32"
    assert mean.dtype == var.dtype == gamma.dtype == beta.dtype == torch.float32, \
        "Mean, var, gamma, and beta tensors must be float32"

    total = batch_size * num_channels * spatial_size
    BLOCK_SIZE = 1024  # Tunable; 256, 512, or 1024 are typical choices

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)

    _triton_kernel_impl[grid](
        input,
        output,
        mean,
        var,
        gamma,
        beta,
        batch_size,
        num_channels,
        spatial_size,
        total,
        BLOCK_SIZE=BLOCK_SIZE
    )
    # Optional synchronization for debugging:
    # torch.cuda.synchronize()